import warnings

# Suppress known false-positive warnings
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*before.*optimizer.step.*')
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

from custom_logger import get_logger
from demo import run
from settings import instance as settings
from training import train
from tuning import tune

logger = get_logger('main')

if __name__ == '__main__':
    config = settings()
    if config.mode == 'train':
        while True:
            if config.tuning:
                logger.info("Tuning mode")
                tune(config)
            logger.info("Training mode")
            if train(config) != -2:
                break
    elif config.mode == 'demo':
        logger.info("Demo mode")
        run(config)
