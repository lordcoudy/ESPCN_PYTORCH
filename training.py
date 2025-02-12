from math import log10

import progress.bar

from utils import *


@measure_time
def test(settings):
    m = settings.model
    m.eval()
    avg_psnr = 0
    max_mse = 0
    min_mse = 1
    with torch.no_grad():
        for batch in settings.testing_data_loader:
            input_tensor, target_tensor = batch[0].to(settings.device), batch[1].to(settings.device)
            output = m(input_tensor)
            mse = settings.criterion(output, target_tensor)
            max_mse = max(max_mse, mse.item())
            min_mse = min(min_mse, mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            print(f"PSNR: {psnr:.6f} dB | MSE: {mse.item():.6f}")
    print(f"===> Avg. PSNR: {avg_psnr / len(settings.testing_data_loader):.12f} dB >===")
    print(f"===> Max. MSE: {max_mse:.12f} >===")
    print(f"===> Min. MSE: {min_mse:.12f} >===")


@measure_time
def train_model(settings):
    m = settings.model
    m.train()
    for epoch in range(settings.epochs_number):
        epoch_loss = 0
        bar = progress.bar.IncrementalBar(f'Epoch {epoch + 1}', max = len(settings.training_data_loader))
        bar.start()
        for iteration, batch in enumerate(settings.training_data_loader, 1):
            data, target = batch[0].to(settings.device), batch[1].to(settings.device)
            bar.next()
            settings.optimizer.zero_grad()
            loss = calculateLoss(settings, data, target)
            epoch_loss += loss.item()
            backPropagate(settings, loss)
            print(f"===> Epoch[{epoch+1}]({iteration}/{len(settings.training_data_loader)}): Loss: {loss.item():.6f}")
        print(f"===> Epoch {epoch+1}/{settings.epochs_number} Complete: Avg. Loss: {epoch_loss / len(settings.training_data_loader):.12f}")

        test(settings)

        bar.finish()
        # Learning rate decay
        if settings.scheduler_enabled:
            settings.scheduler.step(epoch_loss)
        # Checkpoint
        if epoch+1 in [25, 50, 100, 200, 500, 1000, 2000]:
            checkpoint(settings, epoch+1)
            export_model(settings, epoch+1)


def train(settings):
    print(f"===> Upscale factor: {settings.upscale_factor} | Epochs: {settings.epochs_number} >===")
    print("===> Building model >===")
    print("Structure of the model: ", settings.model)
    print(f"{settings.name}: ", end = "\n", file = open(f'times\\time_train_model.txt', 'a+'))
    train_model(settings)
