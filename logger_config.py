
import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "adaptive_rag.log"


def setup_logger():
    logger = logging.getLogger("adaptive_rag")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        )

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
