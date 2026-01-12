import logging
import os

# Global flag to control console verbosity (set by Settings)
_verbose_logging = False


def set_verbose_logging(verbose: bool) -> None:
    """Set global verbose logging flag. Called by Settings during init."""
    global _verbose_logging
    _verbose_logging = verbose
    # Update existing stream handlers
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.INFO if verbose else logging.WARNING)


def _get_log_dir():
    """Get log directory, with fallback to avoid circular import during Settings init."""
    try:
        # Check if Settings instance already exists (avoid creating during init)
        from settings import Settings
        if Settings in Settings._instances:
            # Settings already initialized, safe to use model_dir_i
            from settings import model_dir_i
            return os.path.join(model_dir_i(), 'logs')
    except (RecursionError, RuntimeError, KeyError):
        pass
    
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

        # File handler for persistent logging (always DEBUG level)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{module_name}.log"))
        file_handler.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler for stdout - level controlled by verbose_logging setting
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO if _verbose_logging else logging.WARNING)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger