"""Logging configuration for the mellowgate library.

This module sets up a rich-formatted logger for the mellowgate library,
providing beautiful console output with syntax highlighting and traceback
formatting for enhanced debugging experience.
"""

import logging

from rich.console import Console
from rich.logging import RichHandler

# Initialize rich console for beautiful output formatting
console = Console()

# Configure root logger with RichHandler for enhanced formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# Create mellowgate-specific logger instance
logger = logging.getLogger("mellowgate")
logger.setLevel(logging.INFO)

__all__ = ["logger"]
