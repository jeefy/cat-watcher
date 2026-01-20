"""Logging configuration for Cat Watcher.

Configures structlog for consistent, clean log output.
"""

import logging
import sys
from typing import Literal

import structlog


def configure_logging(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> None:
    """Configure logging for the application.

    Args:
        log_level: Minimum log level to display
    """
    # Map string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,  # Override any existing configuration
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(level)

    # Configure structlog with filtering
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # Allow reconfiguration
    )


# Configure with default INFO level on module import
# This ensures all loggers created before explicit configuration still work
configure_logging("INFO")
