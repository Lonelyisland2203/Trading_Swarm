"""Logging configuration utilities."""

import sys

from loguru import logger


# Standard CLI log format with colors
CLI_LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

# Compact format without function name
CLI_LOG_FORMAT_COMPACT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


def configure_cli_logging(
    level: str = "INFO",
    include_name: bool = True,
) -> None:
    """
    Configure loguru for CLI scripts with consistent formatting.

    Removes default handler and adds a formatted stderr handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        include_name: If True, include module:function in log lines

    Example:
        if __name__ == "__main__":
            configure_cli_logging()
            main()
    """
    logger.remove()
    fmt = CLI_LOG_FORMAT if include_name else CLI_LOG_FORMAT_COMPACT
    logger.add(
        sys.stderr,
        format=fmt,
        level=level.upper(),
    )
