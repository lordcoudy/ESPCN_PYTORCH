# ESPCN_PYTORCH

This project is a modified implementation of the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) [https://doi.org/10.48550/arXiv.1609.05158] in PyTorch.

## Overview
This project implements the ESPCN model for image super-resolution using PyTorch.\
The model is designed to upscale low-resolution images to higher resolutions efficiently.\
For training it uses the BSD300 and BSDS500 datasets.

## Requirements
Described in `requirements.txt`:
- Python 3.x
- PyTorch (w/ CUDA support)
- torchvision
- numpy
- pyyaml
- PIL (Pillow)
- SIX
- optuna
- progress
- click
- fsspec

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/lordcoudy/ESPCN_PYTORCH.git
    cd ESPCN_PYTORCH
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

## Usage
All configurations are made in the `settings.yaml` file.

### Training the Model
To train the model, set the mode to `train` in `settings.yaml`:
```yaml
mode: train
```
Then run:
```commandline
python main.py
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
python main.py
```
## Configuration
The settings.yaml file contains various configuration options:
```yaml
---
input_path: "E:/SAVVA/STUDY/CUDA/TESTS/DATASET/archive/Set5/Set5" # can be a directory with images or a single image
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
learning_rate: 0.0009346227518033354
momentum: 0.9444533942913925
weight_decay: 0.00005079110223222985
threads: 8
optimizer: "adam" # "adam", "sgd"
# Recalculate conditions
psnr_delta: 0
stuck_level: 30
target_min_psnr: 26
# Optimizations
cuda: true
tuning: true
trials: 150 # Tuning trials
mixed_precision: true
optimized: true # Enable Classifier
num_classes: 5 # Number of classes for Classifier
separable: true # Enable separable Conv2d
scheduler: true # Enable learning rate scheduler
pruning: false # Enable pruning
prune_amount: 0.1
preload: true # Preload model
preload_path: "./results/2x_epochs(1500)_optimized(5)_cuda_tuning_mixed_precision_with_scheduler_separable_optimizer(adam)_seed(123)_batch_size(16)_lr(0.0009346227518033354)_momentum(0.9444533942913925)_weight_decay(5.079110223222985e-05)_ckp422.pth"
# Miscellaneous
seed: 123
show_progress_bar: true
show_profiler: true
show_result: false
cycles: 200

```
## Model Export
The model can be exported after training. The exported model will be saved in the specified directory with the naming convention:
```
{upscale_factor}x_epoch_{n_epochs}_optimized({optimized})_cuda({cuda})_tuning({tuning})_pruning({pruning})_mp({mp})_scheduler({scheduler_enabled}).pt
```
## Pruning the Model
The model can be pruned to reduce its size and improve inference speed. The pruning amount can be adjusted in the settings file. 
## Testing the Model
The model is tested using the test function, which evaluates the model on the test dataset and prints the average PSNR, maximum MSE, and minimum MSE.
## License
This project is licensed under the MIT License.
```
## Acknowledgements
This project is inspired by the original ESPCN paper and various implementations available online.
```
