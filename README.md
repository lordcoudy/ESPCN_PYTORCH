# ESPCN_PYTORCH

This project is a modified implementation of the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) [https://doi.org/10.48550/arXiv.1609.05158] in PyTorch.

## Overview
This project implements the ESPCN model for image super-resolution using PyTorch.\
The model is designed to upscale low-resolution images to higher resolutions efficiently.\
For training it uses the BSDS500 dataset.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/lordcoudy/ESPCN_PYTORCH.git
    cd ESPCN_PYTORCH
    ```
2. Install poetry
    ### osx / linux / bashonwindows / Windows+MinGW install instructions

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    ### windows powershell install instructions

    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```


3. Create a virtual environment and install dependencies:
    ```bash
    poetry install
    ```

## Usage
All configurations are made in the [settings.yaml](settings.yaml) file.

### ⚡ Quick Start with Auto-Configuration (Recommended)

**NEW:** Automatically detect your hardware and configure optimal settings:

```bash
# Interactive configuration wizard (recommended for first-time users)
poetry run python autoconfig_cli.py

# Or show recommendations only
poetry run python autoconfig_cli.py --info

# Apply optimized settings directly to settings.yaml
poetry run python autoconfig_cli.py --apply

# Compare your current settings with recommendations
poetry run python autoconfig_cli.py --compare

# Train with auto-configuration
poetry run python main.py --autoconfig
```

The autoconfiguration system detects:
- **CPU**: Physical/logical cores for optimal worker threads
- **RAM**: Available memory for batch size optimization
- **GPU**: CUDA/MPS availability and VRAM for device-specific settings
- **Platform**: OS-specific optimizations (Apple Silicon, CUDA, CPU)

Based on hardware, it automatically configures:
- Optimal batch sizes (train/test)
- DataLoader worker threads
- Device selection (CUDA > MPS > CPU)
- Mixed precision training
- Memory optimizations (channels_last, gradient accumulation)
- Performance features (model compilation, fused optimizers)

### Training the Model
To train the model, set the mode to `train` in [settings.yaml](settings.yaml):
```yaml
mode: train
```
Then run:
```commandline
poetry run python main.py
```
### Run demo
To run the demo, set the mode to demo in settings.yaml:
```yaml
mode: demo
```
**Note:** You should have a .pth model file in the models directory.\
\
Then run:
```commandline
poetry run python main.py
```
## Configuration
The settings.yaml file contains various configuration options:

### Hardware Tiers

The autoconfiguration system classifies your hardware into performance tiers:

| Tier | Requirements | Optimizations |
|------|-------------|---------------|
| **ULTRA** | CUDA GPU (8+ GB VRAM) + 16+ GB RAM | Max batch sizes, torch.compile (max-autotune), FP16, fused optimizers, dataset caching |
| **HIGH** | Any GPU + 8+ GB RAM + 4+ CPU cores | Good batch sizes, torch.compile, mixed precision, persistent workers |
| **MEDIUM** | GPU + 4+ GB RAM OR 8+ GB RAM + 4+ cores | Moderate batch sizes, basic optimizations, gradient accumulation |
| **LOW** | Limited resources | Small batch sizes, gradient accumulation (2x), conservative settings |

### Manual Configuration

If you prefer manual configuration, edit [settings.yaml](settings.yaml):

```yaml
---
input_path: "path/to/images" # can be a directory with images or a single image
output_path: "./results/"
model_path: "./models/"
upscale_factor: 2 # 2, 3, 4, 8
mode: "demo" # "train", "demo"
# Epoch settings
epochs_number: 2000
epoch: 2000
checkpoint_frequency: 100
# Training settings (only for mode: "train"). Do not change
batch_size: 16
test_batch_size: 8
learning_rate: 0.001
momentum: 0.95
weight_decay: 0.00005
threads: 8
optimizer: "adam" # "adam", "sgd"
# Recalculate conditions
psnr_delta: 0
stuck_level: 30
target_min_psnr: 26
# Optimizations
cuda: true
tuning: false
trials: 150 # Tuning trials
mixed_precision: true
optimized: false # Enable Classifier
num_classes: 5 # Number of classes for Classifier
separable: true # Enable separable Conv2d
scheduler: false # Enable learning rate scheduler
pruning: true # Enable pruning
prune_amount: 0.1
preload: false # Preload model
preload_path: "/path/to/pretrained.pth"
# Miscellaneous
seed: 123
show_progress_bar: true
show_profiler: true
show_result: false
cycles: 200

```
## Model Export
The model can be exported after training. The exported model will be saved in the specified directory in .pth and jit traced formats.

## Testing the Model
The model is tested using the test function, which evaluates the model on the test dataset and prints the average PSNR, maximum MSE, and minimum MSE.
## License
This project is licensed under the MIT License.

## Acknowledgements
This project is inspired by the original ESPCN paper and various implementations available online.

