"""
logger.py
---------
Centralised logging setup for the BPS framework.

Two handlers:
  - FileHandler  (DEBUG): output/<run>/debug.log  — always on, full detail
  - StreamHandler (INFO or DEBUG): console — only with verbose=True

Usage:
    from src.logger import setup_logging
    setup_logging(output_dir="output", verbose=True)

Each module then gets its own child logger:
    import logging
    logger = logging.getLogger("bps.senior_clerk")
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path


def setup_logging(output_dir: str = "output", verbose: bool = False) -> None:
    """Configure the root 'bps' logger. Call once at startup."""
    root = logging.getLogger("bps")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # ── File handler: always at DEBUG, full structured format ──────────
    log_path = Path(output_dir) / "debug.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)5s] %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(fh)

    # ── Console handler: only with --verbose ───────────────────────────
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter("    [DBG] %(name)-14s | %(message)s"))
        root.addHandler(ch)
