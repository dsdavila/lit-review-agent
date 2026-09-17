"""filtering.log setup - tracks why papers were rejected and why downloads failed.

Reconfigured once per conference (see main.py): each conference gets its
own filtering.log under its own data/{conference}/ directory, plus a shared
console handler.
"""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(conf_dir: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("lit_review_agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    file_handler = logging.FileHandler(conf_dir / "filtering.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger
