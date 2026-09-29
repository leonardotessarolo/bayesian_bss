import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

class LoggingUtilities:
    @staticmethod
    def configure_logging(
            log_dir: Path,
            log_type: str
        ):
        log_path = log_dir / "{}_logs.log".format(log_type)

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Rotate after 20 MB, preserving five previous files.
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Keep useful progress visible in the terminal.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Avoid duplicated output if this setup function is called twice,
        # which is particularly common in Jupyter notebooks.
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)