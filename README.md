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

