import logging
import os

def get_logger(module_name):
    log_level = "INFO"
    numeric_level = getattr(logging, log_level, logging.DEBUG)

    logger = logging.getLogger(module_name)
    if not logger.handlers:
        logger.setLevel(numeric_level)
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(log_dir, f"{module_name}.log"))
        file_handler.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger