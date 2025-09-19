from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()

# Configure root logger with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("mellowgate")
logger.setLevel(logging.INFO)

__all__ = ["logger"]
