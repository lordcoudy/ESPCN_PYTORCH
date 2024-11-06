# ESPCN_PYTORCH

Simple implementation of ESPCN (Efficient Sub-Pixel Convolutional Neural Network) in PyTorch.

## Overview
This project implements the ESPCN model for image super-resolution using PyTorch.\
The model is designed to upscale low-resolution images to higher resolutions efficiently.\
For training it uses the BSD300 dataset, which contains 300 high-resolution images.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- pyyaml
- PIL (Pillow)

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
### Tuning the Model
To tune the model, set the mode to tune in settings.yaml:
```yaml
mode: tune
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
Then run:
```commandline
python main.py
```
## Configuration
The settings.yaml file contains various configuration options:
```yaml
input_path: '/path/to/input/image.jpg'
output_path: '/path/to/output/directory/'
upscale_factor: 2
batch_size: 16
test_batch_size: 16
epochs_number: 1000
learning_rate: 1e-4
threads: 8
seed: 123
cuda: false
mode: 'train'
trials: 100
show_progress_bar: true
mixed_precision: false
model: 'Optimized'
model_path: '/path/to/model/directory/'
```
## Model Export
The model can be exported after training. The exported model will be saved in the specified directory with the naming convention x_traced_espcn_epoch_<epoch>.pt.
## Pruning the Model
The model can be pruned to reduce its size and improve inference speed. The pruning amount can be adjusted in the pruneModel function. 
## Testing the Model
The model can be tested using the test function, which evaluates the model on the test dataset and prints the average PSNR, maximum MSE, and minimum MSE.
## License
This project is licensed under the MIT License.
```
## Acknowledgements
This project is inspired by the original ESPCN paper and various implementations available online.
```
