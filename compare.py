import torch
import matplotlib.pyplot as plt
import argparse
import os
import re
import matplotlib.image as mpimg
import json

# --- 1. Setup ---

def get_epoch_from_path(filepath):
    match = re.search(r"epoch_(\d+).pth", os.path.basename(filepath))
    if match:
        return int(match.group(1))
    raise ValueError("Could not parse epoch number from checkpoint file. Expected 'epoch_XX.pth'")

comparison_dir = "results/comparison"
os.makedirs(comparison_dir, exist_ok=True)

# --- 2. Parse Arguments ---
parser = argparse.ArgumentParser(description="Compare DCGAN and WGAN-GP models.")
parser.add_argument(
    '--dcgan_ckpt', 
    type=str, 
    required=True, 
    help='Path to the FINAL DCGAN checkpoint (e.g., checkpoints/epoch_49.pth)'
)
parser.add_argument(
    '--wgan_ckpt', 
    type=str, 
    required=True, 
    help='Path to the FINAL WGAN-GP checkpoint (e.g., checkpoints_wgan/epoch_49.pth)'
)
parser.add_argument(
    '--dcgan_fid', 
    type=str, 
    default='dcgan.json',
    help='Path to the DCGAN FID history JSON file.'
)
parser.add_argument(
    '--wgan_fid', 
    type=str, 
    default='wgan.json',
    help='Path to the WGAN-GP FID history JSON file.'
)
args = parser.parse_args()

print("Loading checkpoints for loss plots...")
dcgan_checkpoint = torch.load(args.dcgan_ckpt, map_location=torch.device('cpu'))
wgan_checkpoint = torch.load(args.wgan_ckpt, map_location=torch.device('cpu'))

# Extract loss lists
dcgan_g_losses = dcgan_checkpoint['g_losses']
dcgan_d_losses = dcgan_checkpoint['d_losses']
wgan_g_losses = wgan_checkpoint['g_losses']
wgan_d_losses = wgan_checkpoint['d_losses']

print("Checkpoints loaded. Generating plots...")

# --- 3. Plot 1: Generator Losses ---
plt.figure(figsize=(12, 7))
plt.title("Generator Loss Comparison (Baseline vs. Improved)", fontsize=16)
plot_every = 5 
plt.plot(
    dcgan_g_losses[::plot_every], 
    label="Baseline DCGAN (G-Loss)", 
    linestyle='-', marker='o', alpha=0.8
)
plt.plot(
    wgan_g_losses[::plot_every], 
    label="Improved WGAN-GP (G-Loss)", 
    linestyle='--', marker='s', alpha=0.8
)
plt.xlabel("Iterations (x100)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plot1_path = os.path.join(comparison_dir, "01_generator_loss_comparison.png")
plt.savefig(plot1_path)
print(f"Saved Generator Loss plot to {plot1_path}")
plt.close()

# --- 4. Plot 2: Discriminator/Critic Losses ---
plt.figure(figsize=(12, 7))
plt.title("Discriminator/Critic Loss Comparison (Baseline vs. Improved)", fontsize=16)
plt.plot(
    dcgan_d_losses[::plot_every], 
    label="Baseline DCGAN (D-Loss)", 
    linestyle='-', marker='o', alpha=0.8
)
plt.plot(
    wgan_d_losses[::plot_every], 
    label="Improved WGAN-GP (Critic-Loss)", 
    linestyle='--', marker='s', alpha=0.8
)
plt.xlabel("Iterations (x100)")
plt.ylabel("Loss")
# Auto-scale y-axis
plt.ylim(min(min(wgan_d_losses), min(dcgan_d_losses)), max(max(wgan_d_losses), max(dcgan_d_losses), 5))
plt.legend()
plt.grid(True)
plot2_path = os.path.join(comparison_dir, "02_discriminator_loss_comparison.png")
plt.savefig(plot2_path)
print(f"Saved Discriminator Loss plot to {plot2_path}")
plt.close()

# --- 5. Plot 3: Qualitative Image Comparison ---
try:
    epoch_num = get_epoch_from_path(args.dcgan_ckpt)
    wgan_img_path = f"results/dcgan_cifar/epoch_{epoch_num}.png"
    dcgan_img_path = f"results/wgan_cifar/epoch_{epoch_num}.png"
    
    if not os.path.exists(dcgan_img_path) or not os.path.exists(wgan_img_path):
        print(f"Warning: Could not find images for epoch {epoch_num}. Skipping qualitative plot.")
    else:
        img_dcgan = mpimg.imread(dcgan_img_path)
        img_wgan = mpimg.imread(wgan_img_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(img_dcgan)
        ax1.set_title(f"Baseline DCGAN (Epoch {epoch_num})", fontsize=18)
        ax1.axis('off')
        
        ax2.imshow(img_wgan)
        ax2.set_title(f"Improved WGAN-GP (Epoch {epoch_num})", fontsize=18)
        ax2.axis('off')
        
        plt.tight_layout()
        plot3_path = os.path.join(comparison_dir, "03_qualitative_comparison.png")
        plt.savefig(plot3_path)
        print(f"Saved Qualitative Comparison plot to {plot3_path}")
        plt.close()

except ValueError as e:
    print(f"Warning: {e}. Skipping qualitative plot.")

# --- 6. PLOT 4: FID Score Comparison ---
print("Loading FID history...")
try:
    with open(args.dcgan_fid, 'r') as f:
        dcgan_fid_data = json.load(f)
    with open(args.wgan_fid, 'r') as f:
        wgan_fid_data = json.load(f)

    # Extract data for plotting
    wgan_epochs = [item['epoch'] for item in dcgan_fid_data]
    wgan_fids = [item['fid'] for item in dcgan_fid_data]
    
    dcgan_epochs = [item['epoch'] for item in wgan_fid_data]
    dcgan_fids = [item['fid'] for item in wgan_fid_data]

    plt.figure(figsize=(12, 7))
    plt.title("FID Score Comparison", fontsize=16)
    
    plt.plot(
        dcgan_epochs, 
        dcgan_fids, 
        label="Baseline DCGAN (FID)", 
        linestyle='-', 
        marker='o'
    )
    plt.plot(
        wgan_epochs, 
        wgan_fids, 
        label="Improved WGAN-GP (FID)", 
        linestyle='--', 
        marker='s'
    )
    
    plt.xlabel("Epoch")
    plt.ylabel("Fréchet Inception Distance (FID)")
    plt.legend()
    plt.grid(True)
    plot4_path = os.path.join(comparison_dir, "04_fid_score_comparison.png")
    plt.savefig(plot4_path)
    print(f"Saved FID Score plot to {plot4_path}")
    plt.close()

except FileNotFoundError:
    print(f"Warning: Could not find FID history files '{args.dcgan_fid}' or '{args.wgan_fid}'.")
    print("Skipping FID plot. Run calculate_fid.py to generate these files.")
except Exception as e:
    print(f"An error occurred while plotting FID: {e}. Skipping plot.")

print("Comparison complete. All plots saved in 'results/comparison/'.")