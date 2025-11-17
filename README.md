# WGAN: Wasserstein Generative Adversarial Network

A PyTorch implementation comparing a baseline Deep Convolutional GAN (DCGAN) with an improved Wasserstein GAN with Gradient Penalty (WGAN-GP) to demonstrate stable training on the CIFAR-10 dataset.

This project implements both models from scratch and provides a full pipeline to train, evaluate (using Fréchet Inception Distance), and compare them.

-----

## 🚀 Core Objective

This repository demonstrates the mathematical and practical benefits of the Wasserstein distance (WGAN-GP) over the standard Binary Cross-Entropy (DCGAN) loss for training Generative Adversarial Networks.

  * **DCGAN (Baseline):** Prone to training instability and **mode collapse** when trained on complex datasets like CIFAR-10.
  * **WGAN-GP (Improved):** Solves these issues by using a mathematically sound loss function (Earth-Mover's Distance) and a gradient penalty, resulting in stable training and higher-quality, more diverse image generation.

-----

## 📂 Project Structure

```
WGAN-Wasserstein_Generative_Adversarial_Network/
├── models.py               # Contains all 4 model architectures (DCGAN G/D, WGAN G/C)
├── train_dcgan.py          # Step 1: Train the baseline DCGAN on CIFAR-10
├── train_wgan.py           # Step 3: Train the improved WGAN-GP on CIFAR-10
├── calculate_fid.py        # Post-training script to generate FID scores for all epochs
├── compare.py              # Step 4: Generate all 4 final comparison plots
|
├── dataset_cifar/          # (Will be created) Downloaded CIFAR-10 dataset
|
├── checkpoints_cifar/      # (Will be created) Saved checkpoints for DCGAN
├── checkpoints_wgan_cifar/ # (Will be created) Saved checkpoints for WGAN-GP
|
└── results/
    ├── dcgan_cifar/        # (Will be created) Generated images & loss plot for DCGAN
    ├── wgan_cifar/         # (Will be created) Generated images & loss plot for WGAN-GP
    └── comparison_cifar/   # (Will be created) Final 4 comparison plots
```

-----

## 🛠️ Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/MannShah0312/WGAN-Wasserstein_Generative_Adversarial_Network.git
    cd WGAN-Wasserstein_Generative_Adversarial_Network
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install the required libraries:**

    ```bash
    pip install torch torchvision matplotlib torch-fidelity scipy
    ```

-----

## 🏃‍♂️ How to Run the Full Experiment

Follow these steps in order to reproduce the project results.

**(Note: Training GANs on CIFAR-10 is computationally expensive and will take a significant amount of time, especially without a powerful GPU.)**

### Step 1: Train the Baseline (DCGAN)

This will train the standard DCGAN for 100 epochs, saving a checkpoint and a sample image grid after each epoch.

```bash
python train_dcgan.py
```

  * **Results** will be saved in `results/dcgan_cifar/`.
  * **Checkpoints** will be saved in `checkpoints_cifar/`.

### Step 2: Train the Improved Model (WGAN-GP)

This will train the WGAN-GP for 100 epochs.

```bash
python train_wgan.py
```

  * **Results** will be saved in `results/wgan_cifar/`.
  * **Checkpoints** will be saved in `checkpoints_wgan_cifar/`.

### Step 3: Calculate FID Scores

After training is complete, run this script **twice** to loop through all saved checkpoints and calculate the FID score for each epoch. This will take time.

**1. Calculate for DCGAN:**

```bash
python calculate_fid.py --model_type dcgan --checkpoints_dir ./checkpoints_cifar --output_file dcgan_cifar_fid_history.json
```

**2. Calculate for WGAN-GP:**

```bash
python calculate_fid.py --model_type wgan --checkpoints_dir ./checkpoints_wgan_cifar --output_file wgan_cifar_fid_history.json
```

This will create two `.json` files in your root directory containing the FID history.

### Step 4: Generate Final Comparison Plots

This script loads the final checkpoints and the two FID history files to generate all four comparison plots.

```bash
# Assumes 100 epochs (epoch_99.pth is the last file)
python compare.py --dcgan_ckpt checkpoints_cifar/epoch_99.pth --wgan_ckpt checkpoints_wgan_cifar/epoch_99.pth
```

Your final four plots will be saved in `results/comparison_cifar/`.

-----

### Optional: Resuming Training

Both `train_dcgan.py` and `train_wgan.py` support resuming from a checkpoint. Just pass the path to the checkpoint file you want to resume from.

```bash
# Example: Resuming DCGAN from epoch 20
python train_dcgan.py --load_checkpoint checkpoints_cifar/epoch_19.pth
```

-----

## 📊 Final Results

After running the full pipeline, the `results/comparison_cifar/` folder will contain the four key comparison graphs for your report.

#### 1\. Generator Loss Comparison

*(Shows the stable loss of WGAN-GP vs. the chaotic loss of DCGAN)*

\`\`

#### 2\. Discriminator/Critic Loss Comparison

*(Shows WGAN-GP's critic loss converging, proving it's learning the distance)*

\`\`

#### 3\. Qualitative Image Comparison

*(Visually demonstrates WGAN-GP's ability to avoid mode collapse)*

\`\`

#### 4\. FID Score Comparison

*(Provides quantitative proof that WGAN-GP produces higher-fidelity and more diverse images)*

\`\`