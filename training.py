import time
from math import log10

import numpy as np
import progress.bar

from utils import *

@measure_time
def test(settings):
    settings.model.eval()
    avg_psnr = 0
    max_mse = 0
    min_mse = 1
    with torch.no_grad():
        for batch in settings.testing_data_loader:
            input_tensor, target_tensor = batch[0].to(settings.device), batch[1].to(settings.device)
            output = settings.model(input_tensor)
            mse = settings.criterion(output, target_tensor)
            max_mse = max(max_mse, mse.item())
            min_mse = min(min_mse, mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print(f"===> Avg. PSNR: {avg_psnr / len(settings.testing_data_loader):.12f} dB >===")
    print(f"===> Max. MSE: {max_mse:.12f} >===")
    print(f"===> Min. MSE: {min_mse:.12f} >===")

@measure_time
def train_model(settings):
    model = settings.model
    model.train()
    epoch_losses = np.zeros(settings.epochs_number)
    for epoch in range(settings.epochs_number):
        epoch_loss = 0
        bar = progress.bar.IncrementalBar(f'Epoch {epoch + 1}', max = settings.training_data_loader.__len__())
        bar.start()
        for data, target in settings.training_data_loader:
            data, target = data.to(settings.device), target.to(settings.device)
            bar.next()
            settings.optimizer.zero_grad()
            loss = mixed_precision(settings, data, target)
            epoch_loss += loss.item()
            epoch_losses[epoch] = epoch_loss

        bar.finish()
        # Learning rate decay
        if settings.scheduler_enabled:
            settings.scheduler.step()
        # Checkpoint
        if epoch+1 in [1, 25, 100, 200, 500, 1000, 2000]:
            test(settings)
            checkpoint(settings, epoch+1)
            export_model(settings, epoch+1)
        print(
                f"Epoch {epoch + 1}/{settings.epochs_number}, Loss: {epoch_loss / len(settings.training_data_loader):.6f}")


def train(settings):
    print(f"===> Upscale factor: {settings.upscale_factor} | Epochs: {settings.epochs_number} >===")
    print("===> Building model >===")
    print("Structure of the model: ", settings.model)
    train_model(settings)
    # plt.plot(dictionary['epochs_number'], g_epochLosses)
    # plt.show()
