import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configures a standard logger for the Hydro-Dataset project.
    Writes to both stdout and logs/<name>.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(getattr(logging, log_level.upper()))
        c_handler.setFormatter(fmt)
        logger.addHandler(c_handler)

        # File handler — logs/<name>.log, append mode
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{name}.log"
        f_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        f_handler.setLevel(getattr(logging, log_level.upper()))
        f_handler.setFormatter(fmt)
        logger.addHandler(f_handler)

    return logger
