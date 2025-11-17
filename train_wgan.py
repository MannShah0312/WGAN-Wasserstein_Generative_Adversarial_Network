import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

# Import our NEW CIFAR model definitions
from models import WGANGP_Generator_CIFAR, WGANGP_Critic_CIFAR, weights_init

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse CLI Arguments
parser = argparse.ArgumentParser(description="Train a WGAN-GP on CIFAR-10.")
parser.add_argument(
    '--load_checkpoint', 
    type=str, 
    default=None, 
    help='Optional path to resume from (e.g., checkpoints_wgan_cifar/epoch_10.pth)'
)
args = parser.parse_args()

# Hyperparameters
LEARNING_RATE = 1e-4
BETA1 = 0.0
BETA2 = 0.9
BATCH_SIZE = 128
IMAGE_SIZE = 32 # CIFAR-10 image size
CHANNELS_IMG = 3   # CIFAR-10 is color
NOISE_DIM = 100
NUM_EPOCHS = 100 # More epochs
FEATURES_D = 64
FEATURES_G = 64
CRITIC_ITERATIONS = 5 # Back to 5, needed for stability on complex data
LAMBDA_GP = 10

# Output directories
results_dir = "results/wgan_cifar"
os.makedirs(results_dir, exist_ok=True)
CHECKPOINT_DIR = "checkpoints_wgan_cifar"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
start_epoch = 0

# --- 2. Load Dataset (CIFAR-10) ---
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
dataset = datasets.CIFAR10(root="dataset_cifar/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Initialize Models and Optimizers ---
gen = WGANGP_Generator_CIFAR(NOISE_DIM, CHANNELS_IMG, FEATURES_G).to(device)
critic = WGANGP_Critic_CIFAR(CHANNELS_IMG, FEATURES_D).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

fixed_noise = torch.randn(64, NOISE_DIM, 1, 1).to(device)
G_losses = []
D_losses = []

# --- 4. Checkpoint Save/Load Functions ---
def save_checkpoint(epoch):
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
    print(f"=> Saving checkpoint '{checkpoint_file}'")
    checkpoint = {
        'epoch': epoch + 1,
        'gen_state_dict': gen.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_critic_state_dict': opt_critic.state_dict(),
        'g_losses': G_losses,
        'd_losses': D_losses,
    }
    torch.save(checkpoint, checkpoint_file)

def load_specific_checkpoint(filepath):
    global start_epoch, G_losses, D_losses
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}. Exiting.")
        exit()

    print(f"=> Loading checkpoint '{filepath}'")
    checkpoint = torch.load(filepath)
    
    gen.load_state_dict(checkpoint['gen_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
    opt_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
    start_epoch = checkpoint['epoch']
    G_losses = checkpoint['g_losses']
    D_losses = checkpoint['d_losses']
    
    print(f"=> Resuming training from epoch {start_epoch}")

# --- 5. Gradient Penalty Function ---
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.randn(real_samples.shape[0], 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

# --- 6. Load Checkpoint OR Initialize Weights ---
if args.load_checkpoint:
    load_specific_checkpoint(args.load_checkpoint)
else:
    print("=> No checkpoint specified, starting from scratch.")
    gen.apply(weights_init)
    critic.apply(weights_init)

# --- 7. Training Loop (WGAN-GP) ---
print("Starting Training (CIFAR-10 WGAN-GP)...")
for epoch in range(start_epoch, NUM_EPOCHS):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.shape[0]

        # --- Train Critic ---
        for _ in range(CRITIC_ITERATIONS):
            critic.zero_grad()
            noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
            fake_images = gen(noise)
            
            critic_real = critic(real_images)
            critic_fake = critic(fake_images.detach())
            gp = compute_gradient_penalty(critic, real_images, fake_images)
            
            loss_critic = (
                torch.mean(critic_fake) - torch.mean(critic_real) + LAMBDA_GP * gp
            )
            
            loss_critic.backward()
            opt_critic.step()
        
        # --- Train Generator ---
        gen.zero_grad()
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
        fake_images_for_gen = gen(noise)
        critic_fake_for_gen = critic(fake_images_for_gen)
        
        loss_gen = -torch.mean(critic_fake_for_gen)
        
        loss_gen.backward()
        opt_gen.step()
        
        # --- Log results ---
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] \t"
                f"Loss_D: {loss_critic:.4f}, Loss_G: {loss_gen:.4f}"
            )
            G_losses.append(loss_gen.item())
            D_losses.append(loss_critic.item())

    # --- Save sample images AND checkpoint at end of epoch ---
    with torch.no_grad():
        fake_imgs_grid = gen(fixed_noise)
        torchvision.utils.save_image(
            fake_imgs_grid[:64], 
            f"{results_dir}/epoch_{epoch}.png", 
            normalize=True
        )
    
    save_checkpoint(epoch)

print("Training Complete!")

# --- 8. Plot and Save Loss Curve ---
plt.figure(figsize=(10, 5))
plt.title("Generator and Critic Loss (WGAN-GP - CIFAR-10)")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D (Critic)")
plt.xlabel("Iterations (x100)")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{results_dir}/loss_curve.png")
print(f"Improved model training complete. Check '{results_dir}' and '{CHECKPOINT_DIR}'.")