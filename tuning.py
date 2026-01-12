from __future__ import print_function

import os
from functools import lru_cache
from typing import Any, Dict

import optuna
import torch
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from custom_logger import get_logger
from data import get_training_set, get_validation_set, worker_init_fn
from utils import backPropagate, calculate_ssim, calculateLoss, measure_time

logger = get_logger('tuning')


# Cache datasets to avoid reloading each trial
@lru_cache(maxsize=1)
def _get_cached_train_dataset(upscale_factor: int):
    return get_training_set(upscale_factor=upscale_factor)


@lru_cache(maxsize=1)
def _get_cached_val_dataset(upscale_factor: int):
    return get_validation_set(upscale_factor=upscale_factor)


# Cache DataLoaders by batch_size to reuse workers
_dataloader_cache: Dict[int, DataLoader] = {}
_val_dataloader_cache: Dict[int, DataLoader] = {}

def _get_dataloader(settings, batch_size: int, validation: bool = False) -> DataLoader:
    """Get or create cached DataLoader for given batch_size.
    
    Args:
        settings: Settings object with device configuration
        batch_size: Batch size for the dataloader
        validation: If True, return validation dataloader
    
    Returns:
        Cached or newly created DataLoader
    """
    cache = _val_dataloader_cache if validation else _dataloader_cache
    
    if batch_size not in cache:
        if validation:
            dataset = _get_cached_val_dataset(settings.upscale_factor)
        else:
            dataset = _get_cached_train_dataset(settings.upscale_factor)
        
        # Use 0 workers on MPS to avoid multiprocessing deadlocks on macOS
        num_workers = 0 if settings.device.type == 'mps' else min(4, os.cpu_count() or 1)
        
        cache[batch_size] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=not validation,
            num_workers=num_workers,
            pin_memory=settings.device.type == 'cuda',
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )
    return cache[batch_size]


def train_model_with_pruning(settings, dataloader, val_dataloader, model, optimizer, trial) -> float:
    """Training loop with Optuna pruning support for early stopping bad trials.
    
    Args:
        settings: Settings object
        dataloader: Training dataloader
        val_dataloader: Validation dataloader  
        model: Model to train
        optimizer: Optimizer
        trial: Optuna trial object
    
    Returns:
        Validation loss
    """
    model.train()
    epoch_loss = 0.0
    
    for iteration, batch in enumerate(dataloader, 1):
        data = batch[0].to(settings.device, non_blocking=True)
        target = batch[1].to(settings.device, non_blocking=True)
        
        if settings.channels_last:
            data = data.to(memory_format=torch.channels_last)
            target = target.to(memory_format=torch.channels_last)
        
        optimizer.zero_grad(set_to_none=True)
        loss = calculateLoss(settings, data, target, model)
        epoch_loss += loss.item()
        backPropagate(settings, loss, optimizer, max_grad_norm=1.0)
        
        # Report intermediate loss for pruning (every 4 batches to reduce overhead)
        if iteration % 4 == 0:
            intermediate_loss = epoch_loss / iteration
            trial.report(intermediate_loss, iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for batch in val_dataloader:
            data = batch[0].to(settings.device, non_blocking=True)
            target = batch[1].to(settings.device, non_blocking=True)
            
            if settings.channels_last:
                data = data.to(memory_format=torch.channels_last)
                target = target.to(memory_format=torch.channels_last)
            
            output = model(data)
            val_loss += settings.criterion(output, target).item()
    
    return val_loss / len(val_dataloader)


def objective(trial: optuna.Trial, settings) -> float:
    """Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        settings: Settings object
    
    Returns:
        Validation loss (lower is better)
    """
    # Hyperparameter search space
    total_trials = getattr(settings, 'trials', None)
    if total_trials:
        message = f"Tuning trial {trial.number + 1}/{total_trials}"
    else:
        message = f"Tuning trial {trial.number + 1}"

    logger.info(message)
    print(message, flush=True)

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    
    # Get cached dataloaders
    train_dataloader = _get_dataloader(settings, batch_size, validation=False)
    val_dataloader = _get_dataloader(settings, batch_size, validation=True)
    
    # Create model with suggested dropout
    from model import ESPCN
    model = ESPCN(
        upscale_factor=settings.upscale_factor,
        num_channels=1,
        separable=settings.separable,
        dropout_rate=dropout_rate
    )
    
    if settings.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.to(settings.device)
    
    # Create optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train with pruning support
    final_loss = train_model_with_pruning(
        settings, train_dataloader, val_dataloader, model, optimizer, trial
    )
    
    return final_loss


@measure_time
def tune(settings) -> Dict[str, Any]:
    """Run hyperparameter tuning with Optuna.
    
    Args:
        settings: Settings object to update with best hyperparameters
    
    Returns:
        Dictionary of best hyperparameters
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Use TPE sampler (smarter than random) + MedianPruner (kills bad trials early)
    sampler = TPESampler(seed=settings.seed, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=4, interval_steps=4)
    
    study = optuna.create_study(
        direction="minimize",
        study_name="ESPCN_tuning",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective(trial, settings),
        n_trials=settings.trials,
        show_progress_bar=settings.show_progress_bar,
        gc_after_trial=True  # Free memory after each trial
    )
    
    # Clear dataloader caches after tuning
    _dataloader_cache.clear()
    _val_dataloader_cache.clear()

    # Log and apply best hyperparameters
    logger.info('Best trial:')
    best_trial = study.best_trial
    logger.info(f"Value: {best_trial.value:.6f}")
    logger.info('Best params:')
    
    best_params = best_trial.params
    for key, value in best_params.items():
        logger.info(f'  {key}: {value}')
    
    # Apply best hyperparameters to settings
    settings.learning_rate = best_params['lr']
    settings.batch_size = best_params['batch_size']
    settings.weight_decay = best_params['weight_decay']
    
    # Recreate optimizer with best params
    optimizer_name = best_params['optimizer']
    if optimizer_name == 'Adam':
        settings._optimizer = optim.Adam(
            settings.model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay
        )
    else:
        settings._optimizer = optim.AdamW(
            settings.model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay
        )
    settings.optimizer_type = optimizer_name
    
    # Recreate the scheduler with the new optimizer after tuning
    if settings.scheduler_enabled:
        settings._optimizer._step_count = 1
        settings._scheduler = optim.lr_scheduler.OneCycleLR(
            settings._optimizer,
            max_lr=settings.learning_rate,
            steps_per_epoch=len(settings.training_data_loader),
            epochs=settings.epochs_number,
            anneal_strategy='cos',
            final_div_factor=1e4
        )
        settings._scheduler._step_count = 1
        logger.info("Scheduler recreated with tuned parameters")
    
    logger.info("Tuning complete, ready for training")
    return best_params