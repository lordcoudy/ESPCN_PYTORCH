from math import log10

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
            mse = calculateLoss(settings, input_tensor, target_tensor)
            max_mse = max(max_mse, mse.item())
            min_mse = min(min_mse, mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            print(f"PSNR: {psnr:.6f} dB | MSE: {mse.item():.6f}")
    avg_psnr /= len(settings.testing_data_loader)
    print(f"===> Avg. PSNR: {avg_psnr:.12f} dB >===")
    print(f"===> Max. MSE: {max_mse:.12f} >===")
    print(f"===> Min. MSE: {min_mse:.12f} >===")
    return

@measure_time
def train_model(settings):
    settings.model.train()
    for epoch in range(settings.epochs_number):
        epoch_loss = 0
        epoch_val_loss = 0
        bar = progress.bar.IncrementalBar(f'Epoch {epoch + 1}', max=len(settings.training_data_loader) + len(settings.validation_data_loader)) # Update bar max
        bar.start()
        for iteration, batch in enumerate(settings.training_data_loader, 1):
            data, target = batch[0].to(settings.device), batch[1].to(settings.device)
            bar.next()
            settings.optimizer.zero_grad()
            loss = calculateLoss(settings, data, target)
            epoch_loss += loss.item()
            backPropagate(settings, loss)
            print(f"===> Epoch[{epoch+1}]({iteration}/{len(settings.training_data_loader)}): Loss: {loss.item():.6f}")

            if settings.scheduler_enabled:
                settings.scheduler.step(epoch_val_loss)

        settings.model.eval()
        with torch.no_grad():
            for val_iteration, val_batch in enumerate(settings.validation_data_loader, 1):
                val_data, val_target = val_batch[0].to(settings.device), val_batch[1].to(settings.device)
                val_loss = calculateLoss(settings, val_data, val_target)
                epoch_val_loss += val_loss.item()
                bar.next()
        settings.model.train()
        print(f"===> Epoch {epoch+1}/{settings.epochs_number} Complete: Avg. Loss: {epoch_loss / len(settings.training_data_loader):.12f} Avg. Val. Loss: {epoch_val_loss / len(settings.validation_data_loader)}")
        bar.finish()
        test(settings)

        if settings.pruning and (epoch + 1) % 100 == 0:
            prune_model(settings.model, settings.prune_amount)

        if (epoch+1) % settings.checkpoint_frequency == 0:
            checkpoint(settings, settings.model, epoch+1)
            export_model(settings, settings.model, epoch+1)


def train(settings):
    print(f"===> Upscale factor: {settings.upscale_factor} | Epochs: {settings.epochs_number} >===")
    print(f"===> Batch size: {settings.batch_size} | Learning rate: {settings.learning_rate} >===")
    print("===> Building model >===")
    print("Structure of the model: ", settings.model)
    os.makedirs('times', exist_ok=True)
    with open(os.path.join('times', 'time_train_model.txt'), 'a+') as f:
        print(f"{settings.name}: ", end="\n", file=f)
    train_model(settings)
