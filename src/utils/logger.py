import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logger(name: str = "lumina"):
    logger = logging.getLogger(name)
    log_handler = logging.StreamHandler(sys.stdout)
    
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    log_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(log_handler)
    
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()
