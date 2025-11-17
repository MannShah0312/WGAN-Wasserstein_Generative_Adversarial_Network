import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch_fidelity
import argparse
import os
import glob
import re
import json
import torch.multiprocessing as mp

# Import our NEW CIFAR models
from models import Generator_CIFAR, WGANGP_Generator_CIFAR

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Helper Functions (Defined globally for pickling) ---
def to_uint8(x):
    # Already [0, 1] from ToTensor()
    return x.mul(255).clamp(0, 255).byte()

class LabelDroppingDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return image

class SimpleTensorDataset(Dataset):
    def __init__(self, tensor_list):
        self.items = tensor_list

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# --- 3. Checkpoint Parsing ---
def get_epoch_from_path(filepath):
    match = re.search(r"epoch_(\d+).pth", os.path.basename(filepath))
    if match:
        return int(match.group(1))
    return -1

# --- 4. Main Function ---
def calculate_fid_history(model_type, checkpoint_dir, output_file):
    print(f"Using device: {device}")
    
    # --- Load Real Dataset (CIFAR-10) ---
    print("Loading real dataset (CIFAR-10)...")
    transform = transforms.Compose([
        transforms.Resize(32), # Ensure 32x32
        transforms.ToTensor(), # To [0, 1]
        transforms.Lambda(to_uint8) # To [0, 255]
    ])
    real_dataset = datasets.CIFAR10(root="dataset_cifar/", train=True, transform=transform, download=True)
    
    # We use 10,000 real images
    real_subset_base = torch.utils.data.Subset(real_dataset, range(10000))
    real_subset = LabelDroppingDataset(real_subset_base)

    # --- Find Checkpoints ---
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if not checkpoint_files:
        print(f"Error: No checkpoints found in {checkpoint_dir}")
        return

    checkpoint_files.sort(key=get_epoch_from_path)
    print(f"Found {len(checkpoint_files)} checkpoints to process.")

    # --- Loop and Calculate ---
    fid_history = []
    
    # Init model architecture
    if model_type == 'dcgan':
        gen = Generator_CIFAR(noise_dim=100, channels_img=3, features_g=64).to(device)
        state_dict_key = 'gen_state_dict'
    elif model_type == 'wgan':
        gen = WGANGP_Generator_CIFAR(noise_dim=100, channels_img=3, features_g=64).to(device)
        state_dict_key = 'gen_state_dict'
    else:
        raise ValueError("model_type must be 'dcgan' or 'wgan'")

    for ckpt_file in checkpoint_files:
        epoch_num = get_epoch_from_path(ckpt_file)
        
        checkpoint = torch.load(ckpt_file, map_location=device)
        gen.load_state_dict(checkpoint[state_dict_key])
        gen.eval()

        # --- PRE-GENERATE IMAGES ---
        print(f"Generating 10,000 fake images for Epoch {epoch_num}...")
        fake_images_list = []
        batch_size_gen = 100 
        
        with torch.no_grad():
            for _ in range(10000 // batch_size_gen):
                noise = torch.randn(batch_size_gen, 100, 1, 1).to(device)
                fake_batch = gen(noise) # Output is [-1, 1]
                
                # Process batch: Denormalize -> [0,1] -> [0,255]
                fake_batch = (fake_batch * 0.5) + 0.5
                fake_batch = fake_batch.mul(255).clamp(0, 255).byte()
                # NO repeat needed, already 3 channels
                
                fake_images_list.append(fake_batch.cpu())
        
        all_fake_images = torch.cat(fake_images_list, dim=0)
        fake_dataset = SimpleTensorDataset(all_fake_images)

        print(f"Calculating FID for Epoch {epoch_num}...")
        
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=real_subset,
            input2=fake_dataset,
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        
        fid_score = metrics_dict['frechet_inception_distance']
        print(f"Epoch {epoch_num}: FID = {fid_score:.4f}")
        fid_history.append({'epoch': epoch_num, 'fid': fid_score})

    # --- Save results to JSON ---
    with open(output_file, 'w') as f:
        json.dump(fid_history, f, indent=4)
    print(f"Saved FID history to {output_file}")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Calculate FID history from checkpoints.")
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True, 
        choices=['dcgan', 'wgan'],
        help="Type of model to evaluate ('dcgan' or 'wgan')."
    )
    parser.add_argument(
        '--checkpoints_dir', 
        type=str, 
        required=True, 
        help="Directory containing the .pth checkpoint files."
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True, 
        help="Path to save the final FID history JSON file."
    )
    
    args = parser.parse_args()
    calculate_fid_history(args.model_type, args.checkpoints_dir, args.output_file)