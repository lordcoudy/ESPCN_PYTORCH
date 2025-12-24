import logging
import os


def _get_log_dir():
    """Get log directory, with fallback to avoid circular import during Settings init."""
    try:
        from settings import model_dir_i
        return os.path.join(model_dir_i(), 'logs')
    except RecursionError:
        # Fallback during Settings initialization
        return os.path.join(os.path.dirname(__file__), 'logs')


def get_logger(module_name):
    log_level = "DEBUG"
    numeric_level = getattr(logging, log_level, logging.DEBUG)

    logger = logging.getLogger(module_name)
    if not logger.handlers:
        logger.setLevel(numeric_level)
        log_dir = _get_log_dir()
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(log_dir, f"{module_name}.log"))
        file_handler.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger