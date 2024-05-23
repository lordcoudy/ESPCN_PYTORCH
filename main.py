from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ESPCN
# from model_ench import EnhancedESPCN as ESPCN
from data import get_training_set, get_test_set

from PIL import Image
from torchvision.transforms import ToTensor

def prepare_data():
    print('===> Loading datasets')
    train_set = get_training_set(upscale_factor)
    test_set = get_test_set(upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=testBatchSize, shuffle=False)
    print('===> Datasets loaded')
    return training_data_loader, testing_data_loader

def train(epoch, training_data_loader):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}/{}]({}/{}): Loss: {:.4f}".format(epoch, nEpochs, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test(testing_data_loader):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch, upscale_factor):
    model_out_path = str(upscale_factor) + "x_espcn_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def export_model(epoch, upscale_factor):
    model.eval()
    input_image = 'E:/SAVVA/STUDY/CUDA/ESPCN-PY/pythonProject/dataset/BSDS300/images/test/3096.jpg'
    img = Image.open(input_image).convert('YCbCr')
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0]).cuda()

    traced_script_module = torch.jit.trace(model, input)
    traced_model_out_path = str(upscale_factor) + "x_traced_espcn_epoch_{}.pt".format(epoch)
    traced_script_module.save(traced_model_out_path)
    print("===> Model exported")
    print("===> Traced model saved to {}".format(traced_model_out_path))


if __name__ == '__main__':
    upscale_factors = [ 2, 3, 4, 8 ]
    batchSize = 16                  # from tuning
    testBatchSize = 8
    arEpochs = [ 25, 50, 100, 200, 500, 1000, 2000 ]
    lr = 0.0009067700037410958      # from tuning
    cuda = True
    threads = 8
    seed = 123

    if cuda and not torch.cuda.is_available():
        raise Exception("===> No GPU found, please run without cuda")

    torch.manual_seed(seed)

    if cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")


    for upscale_factor in upscale_factors:
        for nEpochs in arEpochs:
            print("===> Upscale factor: {} | Epochs: {}".format(upscale_factor, nEpochs))
            training_data_loader, testing_data_loader = prepare_data()
            print("===> Building model")
            model = ESPCN(upscale_factor=upscale_factor).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(1, nEpochs + 1):
                train(epoch, training_data_loader)
                test(testing_data_loader)
            checkpoint(nEpochs, upscale_factor)
            export_model(nEpochs, upscale_factor)
