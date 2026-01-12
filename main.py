import argparse
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
    parser = argparse.ArgumentParser(description='ESPCN Super-Resolution Training')
    parser.add_argument(
        '--autoconfig',
        action='store_true',
        help='Run hardware detection and auto-configure optimal settings before training'
    )
    parser.add_argument(
        '--autoconfig-only',
        action='store_true',
        help='Only run autoconfiguration and exit (useful for testing)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup when applying autoconfig (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Handle autoconfiguration
    if args.autoconfig or args.autoconfig_only:
        logger.info("Running hardware autoconfiguration...")
        from autoconfig import detect_and_configure
        
        detect_and_configure(
            apply=True,
            print_info=True,
            save_to=None,
            upscale_factor=None
        )
        
        if args.autoconfig_only:
            logger.info("Autoconfiguration complete. Exiting.")
            exit(0)
        
        logger.info("Autoconfiguration applied. Starting training with optimized settings...")
    
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
