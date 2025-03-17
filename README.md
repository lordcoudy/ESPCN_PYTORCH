# ESPCN_PYTORCH

This project is a modified implementation of the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) [
https://doi.org/10.48550/arXiv.1609.05158] in PyTorch.

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
input_path: "./input/"
output_path: "./results/"
model_path: "./opt-models/"
upscale_factor: 2 # 2, 3, 4, 8
mode: "train" # "train" or "demo"
# Epoch settings
epochs_number: 1000
epoch: 1000
checkpoint_frequency: 50
# Training settings (only for mode: "train"). Do not change
batch_size: 32
test_batch_size: 8
learning_rate: 0.0001
momentum: 0.9
weight_decay: 0.0001
threads: 8
psnr_delta: 0.0001
stuck_level: 5
target_min_psnr: 25
seed: 123
num_classes: 4
# Optimizations
cuda: true
tuning: false
mixed_precision: false
optimized: false
separable: false
scheduler: false
pruning: false
preload: false
preload_path: "./path/to/model.pth"
# Miscellaneous
trials: 250
show_progress_bar: true
prune_amount: 0.1
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
