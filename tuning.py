from __future__ import print_function

import os
from functools import lru_cache

import optuna
import torch
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from custom_logger import get_logger
from data import get_training_set, worker_init_fn
from utils import backPropagate, calculateLoss, measure_time

logger = get_logger('tuning')

# Cache dataset to avoid reloading each trial
@lru_cache(maxsize=1)
def _get_cached_dataset(upscale_factor):
    return get_training_set(upscale_factor=upscale_factor)

# Cache DataLoaders by batch_size to reuse workers
_dataloader_cache = {}

def _get_dataloader(settings, batch_size):
    """Get or create cached DataLoader for given batch_size."""
    if batch_size not in _dataloader_cache:
        train_set = _get_cached_dataset(settings.upscale_factor)
        # Use 0 workers on MPS to avoid multiprocessing deadlocks on macOS
        num_workers = 0 if settings.device.type == 'mps' else min(4, os.cpu_count() or 1)
        _dataloader_cache[batch_size] = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=settings.device.type == 'cuda',
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )
    return _dataloader_cache[batch_size]


def train_model_with_pruning(settings, dataloader, model, optimizer, trial):
    """Training loop with Optuna pruning support for early stopping bad trials."""
    epoch_loss = 0
    for iteration, batch in enumerate(dataloader, 1):
        data = batch[0].to(settings.device, non_blocking=True)
        target = batch[1].to(settings.device, non_blocking=True)
        if settings.channels_last:
            data = data.to(memory_format=torch.channels_last)
            target = target.to(memory_format=torch.channels_last)
        
        optimizer.zero_grad(set_to_none=True)
        loss = calculateLoss(settings, data, target, model)
        epoch_loss += loss.item()
        backPropagate(settings, loss, optimizer)
        
        # Report intermediate loss for pruning (every 4 batches to reduce overhead)
        if iteration % 4 == 0:
            intermediate_loss = epoch_loss / iteration
            trial.report(intermediate_loss, iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return epoch_loss / len(dataloader)


def objective(trial, settings):
    # Hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)  # Narrower, more practical range
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Remove 128 (often OOM/slow)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True)  # Narrower range
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])  # AdamW often better
    
    # Get cached dataloader (no recreation overhead)
    dataloader = _get_dataloader(settings, batch_size)
    
    # Create model (lightweight for ESPCN)
    model = settings.create_model()
    if settings.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.to(settings.device)
    model.train()
    
    # Create optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train with pruning support
    final_loss = train_model_with_pruning(settings, dataloader, model, optimizer, trial)
    return final_loss


@measure_time
def tune(settings):
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
    
    # Clear dataloader cache after tuning
    _dataloader_cache.clear()

    logger.info('Best trial:')
    trial = study.best_trial
    logger.info(f"Value: {trial.value:.6f}")
    logger.info('Params: ')
    for key, value in trial.params.items():
        logger.info(f'  {key}: {value}')
        if key == 'lr':
            settings.learning_rate = value
        if key == 'batch_size':
            settings.batch_size = value
        if key == 'weight_decay':
            settings.weight_decay = value
        if key == 'optimizer':
            if value == 'Adam':
                settings.optimizer = optim.Adam(settings.model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
            else:
                settings.optimizer = optim.AdamW(settings.model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)