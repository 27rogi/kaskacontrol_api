import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console, markup=True)],
    )
    return logging.getLogger("app")
