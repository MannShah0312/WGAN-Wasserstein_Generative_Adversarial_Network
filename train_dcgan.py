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
from models import Generator_CIFAR, Discriminator_CIFAR, weights_init

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse CLI Arguments
parser = argparse.ArgumentParser(description="Train a DCGAN on CIFAR-10.")
parser.add_argument(
    '--load_checkpoint', 
    type=str, 
    default=None, 
    help='Optional path to resume from (e.g., checkpoints_cifar/epoch_10.pth)'
)
args = parser.parse_args()

# Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 32 # CIFAR-10 image size
CHANNELS_IMG = 3   # CIFAR-10 is color
NOISE_DIM = 100
NUM_EPOCHS = 100 # More epochs needed for CIFAR-10
FEATURES_D = 64
FEATURES_G = 64

# Output directory for results
results_dir = "results/dcgan_cifar"
os.makedirs(results_dir, exist_ok=True)

# Checkpoint setup
CHECKPOINT_DIR = "checkpoints_cifar"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
start_epoch = 0

# --- 2. Load Dataset (CIFAR-10) ---
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        # Normalize to [-1, 1] for 3 channels
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
dataset = datasets.CIFAR10(root="dataset_cifar/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Initialize Models, Loss, Optimizers ---
gen = Generator_CIFAR(NOISE_DIM, CHANNELS_IMG, FEATURES_G).to(device)
disc = Discriminator_CIFAR(CHANNELS_IMG, FEATURES_D).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()
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
        'disc_state_dict': disc.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_disc_state_dict': opt_disc.state_dict(),
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
    disc.load_state_dict(checkpoint['disc_state_dict'])
    opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
    opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
    start_epoch = checkpoint['epoch']
    G_losses = checkpoint['g_losses']
    D_losses = checkpoint['d_losses']
    
    print(f"=> Resuming training from epoch {start_epoch}")

# --- 5. Load Checkpoint OR Initialize Weights ---
if args.load_checkpoint:
    load_specific_checkpoint(args.load_checkpoint)
else:
    print("=> No checkpoint specified, starting from scratch.")
    gen.apply(weights_init)
    disc.apply(weights_init)

# --- 6. Training Loop ---
print("Starting Training (CIFAR-10)...")
for epoch in range(start_epoch, NUM_EPOCHS):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.shape[0]

        # --- Train Discriminator ---
        disc.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        output_real = disc(real_images)
        lossD_real = criterion(output_real, real_labels)
        lossD_real.backward()

        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
        fake_images = gen(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        output_fake = disc(fake_images.detach()) 
        lossD_fake = criterion(output_fake, fake_labels)
        lossD_fake.backward()
        
        lossD = lossD_real + lossD_fake
        opt_disc.step()

        # --- Train Generator ---
        gen.zero_grad()
        output_fool = disc(fake_images)
        lossG = criterion(output_fool, real_labels)
        lossG.backward()
        opt_gen.step()

        # --- Log results ---
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] \t"
                f"Loss_D: {lossD:.4f}, Loss_G: {lossG:.4f}"
            )
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

    # --- Save sample images AND checkpoint at end of epoch ---
    with torch.no_grad():
        fake_imgs_grid = gen(fixed_noise)
        torchvision.utils.save_image(
            fake_imgs_grid[:64], 
            f"{results_dir}/epoch_{epoch}.png", 
            normalize=True # Converts [-1, 1] to [0, 1] for saving
        )
    
    save_checkpoint(epoch)

print("Training Complete!")

# --- 7. Plot and Save Loss Curve ---
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss (Baseline DCGAN - CIFAR-10)")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations (x100)")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{results_dir}/loss_curve.png")
print(f"Baseline training complete. Check '{results_dir}' and '{CHECKPOINT_DIR}'.")