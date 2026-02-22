# src/llmcore/logging_config.py
"""
Unified Logging Configuration for the LLMCore Ecosystem.

This module provides a centralized logging setup that can be used by all
components (llmcore, llmchat, semantiscan, confy) to ensure consistent
logging behavior across the entire application stack.

Configuration is read from confy (the single source of truth) and supports:
- Console logging with display-level gating (see DisplayFilter)
- File logging with configurable directory, filename patterns, and rotation
- Per-component log level overrides
- Consistent formatting across all components

Key concepts:

    **Display filter**: When ``console_enabled=False`` (the default), the
    console handler still exists but only passes through log records that
    carry ``extra={"display": True}``.  This lets operational messages
    (e.g. "Indexing 42 files...") reach the user even in "silent" mode,
    while normal debug/info chatter stays file-only.

    **File rotation**: ``file_mode="single"`` uses a
    ``RotatingFileHandler`` with configurable max size and backup count.
    The default ``file_mode="per_run"`` creates a new timestamped file
    each invocation (unchanged from previous behavior).

Usage:
    from llmcore.logging_config import configure_logging, log_display

    # Early in application startup (before other imports that use logging):
    configure_logging(app_name="llmchat")

    # Log a message that always reaches the console:
    import logging
    logger = logging.getLogger("llmchat.startup")
    log_display(logger, logging.INFO, "Ready - listening on port %d", port)
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# Try to import confy for config loading
try:
    from confy import Confy

    CONFY_AVAILABLE = True
except ImportError:
    CONFY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default logging configuration
# ---------------------------------------------------------------------------
# NOTE: console_enabled defaults to False.  The console handler is still
# created (for display=True records) but normal records are blocked by
# DisplayFilter.  Set console_enabled=True for verbose/debug sessions.
# ---------------------------------------------------------------------------

DEFAULT_LOGGING_CONFIG: dict[str, Any] = {
    "console_enabled": False,
    "console_level": "WARNING",
    "console_format": "%(levelname)s - %(message)s",
    "file_enabled": True,
    "file_level": "DEBUG",
    "file_directory": "~/.local/share/llmcore/logs",
    "file_mode": "per_run",
    "file_name_pattern": "{app}_{timestamp:%Y%m%d_%H%M%S}.log",
    "file_single_name": "{app}.log",
    "file_format": "%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)",
    "rotation_max_bytes": 10 * 1024 * 1024,  # 10 MB
    "rotation_backup_count": 5,
    "display_min_level": "INFO",
    "components": {
        "llmchat": "INFO",
        "llmcore": "INFO",
        "semantiscan": "INFO",
        "confy": "WARNING",
        "urllib3": "WARNING",
        "httpx": "WARNING",
        "httpcore": "WARNING",
        "asyncio": "WARNING",
        "aiosqlite": "WARNING",
    },
}


# ---------------------------------------------------------------------------
# DisplayFilter
# ---------------------------------------------------------------------------


class DisplayFilter(logging.Filter):
    """Controls which log records pass through to the console handler.

    Behavior matrix::

        +------------------------+--------------+-----------------+
        | console_global         | display=True | display=False/  |
        |                        |              | absent          |
        +------------------------+--------------+-----------------+
        | True  (-v mode)        | PASS         | PASS            |
        | False (default/quiet)  | PASS*        | BLOCK           |
        +------------------------+--------------+-----------------+

        * subject to display_min_level

    When console is globally enabled (``-v`` / verbose), everything passes
    and the handler's own level check does the filtering.

    When console is globally disabled (default), **only** records with
    ``record.display = True`` pass through, provided they also meet the
    ``display_min_level`` threshold.  This allows operational messages
    like "Indexing 42 files..." to appear even in silent mode.

    The ``display=True`` flag does **not** bypass the minimum display
    level.  A DEBUG record with ``display=True`` will not show on console
    unless ``display_min_level`` is DEBUG.
    """

    def __init__(
        self,
        console_globally_enabled: bool = False,
        display_min_level: int = logging.INFO,
    ) -> None:
        """Initialize the display filter.

        Args:
            console_globally_enabled: If True, pass all records (verbose mode).
            display_min_level: Minimum level for ``display=True`` records.
                Even with ``display=True``, records below this level are
                blocked.
        """
        super().__init__()
        self.console_globally_enabled = console_globally_enabled
        self.display_min_level = display_min_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if the record should pass to console."""
        if self.console_globally_enabled:
            return True  # Verbose mode: handler-level filtering suffices

        # Silent mode: only display=True records above min level
        if getattr(record, "display", False):
            return record.levelno >= self.display_min_level

        return False


# ---------------------------------------------------------------------------
# UnifiedLoggingManager
# ---------------------------------------------------------------------------


class UnifiedLoggingManager:
    """
    Singleton manager for unified logging configuration.

    Ensures logging is only configured once and provides methods
    for runtime adjustment of log levels.
    """

    _instance: Optional["UnifiedLoggingManager"] = None
    _configured: bool = False
    _log_file_path: Path | None = None
    _console_handler: logging.Handler | None = None
    _file_handler: logging.Handler | None = None
    _display_filter: DisplayFilter | None = None

    def __new__(cls) -> "UnifiedLoggingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "UnifiedLoggingManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_configured(cls) -> bool:
        """Check if logging has been configured."""
        return cls._configured

    @classmethod
    def get_log_file_path(cls) -> Path | None:
        """Get the current log file path."""
        return cls._log_file_path

    def configure(
        self,
        app_name: str = "llmcore",
        config: dict[str, Any] | None = None,
        config_file_path: str | Path | None = None,
        force_reconfigure: bool = False,
    ) -> Path:
        """
        Configure unified logging for the application.

        Args:
            app_name: Name of the application (used in log filename)
            config: Pre-loaded configuration dictionary (logging section)
            config_file_path: Path to config file (if config not provided)
            force_reconfigure: If True, reconfigure even if already configured

        Returns:
            Path to the log file (if file logging enabled)
        """
        if self._configured and not force_reconfigure:
            return self._log_file_path or Path("/dev/null")

        # Load configuration
        log_config = self._load_config(config, config_file_path)

        # Clear any existing handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        # Set root logger to capture everything
        root_logger.setLevel(logging.DEBUG)

        # --- Console handler (always created, gated by DisplayFilter) ---
        console_globally_enabled = log_config.get("console_enabled", False)

        # Resolve display_min_level
        display_min_level_str = log_config.get("display_min_level", "INFO").upper()
        display_min_level = logging.getLevelName(display_min_level_str)
        if not isinstance(display_min_level, int):
            display_min_level = logging.INFO

        self._display_filter = DisplayFilter(
            console_globally_enabled=console_globally_enabled,
            display_min_level=display_min_level,
        )

        # Always create console handler so display=True records can appear
        # even when console_enabled=False.  Wrap in try/except for
        # environments without stderr (R4 risk mitigation).
        try:
            self._console_handler = self._create_console_handler(log_config)
            if not console_globally_enabled:
                # Console is "off" - set handler level to DEBUG so the
                # filter is the sole gate (display=True passes, rest blocked).
                self._console_handler.setLevel(logging.DEBUG)
            self._console_handler.addFilter(self._display_filter)
            root_logger.addHandler(self._console_handler)
        except Exception:
            # Degrade gracefully if stderr is unavailable
            self._console_handler = None

        # --- File handler ---
        if log_config.get("file_enabled", True):
            self._file_handler, self._log_file_path = self._create_file_handler(
                log_config, app_name
            )
            if self._file_handler:
                root_logger.addHandler(self._file_handler)

        # Configure per-component log levels
        components = log_config.get("components", DEFAULT_LOGGING_CONFIG["components"])
        for component_name, level_str in components.items():
            component_logger = logging.getLogger(component_name)
            level = logging.getLevelName(level_str.upper())
            if isinstance(level, int):
                component_logger.setLevel(level)

        UnifiedLoggingManager._configured = True
        UnifiedLoggingManager._log_file_path = self._log_file_path

        # Log startup message to file only (not console)
        if self._log_file_path:
            file_logger = logging.getLogger("llmcore.logging_config")
            file_logger.debug(f"Unified logging configured. Log file: {self._log_file_path}")

        return self._log_file_path or Path("/dev/null")

    def _load_config(
        self, config: dict[str, Any] | None, config_file_path: str | Path | None
    ) -> dict[str, Any]:
        """Load logging configuration from various sources."""

        # If config provided directly, use it
        if config is not None:
            return {**DEFAULT_LOGGING_CONFIG, **config}

        # Try to load from confy
        if CONFY_AVAILABLE:
            try:
                confy = Confy(
                    app_name="llmcore",
                    config_file_path=str(config_file_path) if config_file_path else None,
                )
                logging_section = confy.get("logging", {})
                if logging_section:
                    return {**DEFAULT_LOGGING_CONFIG, **logging_section}
            except Exception:
                pass  # Fall through to defaults

        return DEFAULT_LOGGING_CONFIG.copy()

    def _create_console_handler(self, config: dict[str, Any]) -> logging.Handler:
        """Create and configure the console handler."""
        handler = logging.StreamHandler(sys.stderr)

        level_str = config.get("console_level", "WARNING").upper()
        level = logging.getLevelName(level_str)
        if isinstance(level, int):
            handler.setLevel(level)
        else:
            handler.setLevel(logging.WARNING)

        fmt = config.get("console_format", DEFAULT_LOGGING_CONFIG["console_format"])
        handler.setFormatter(logging.Formatter(fmt))

        return handler

    def _create_file_handler(
        self, config: dict[str, Any], app_name: str
    ) -> tuple[logging.Handler | None, Path | None]:
        """Create and configure the file handler.

        Supports two modes:

        - ``"per_run"`` (default): creates a new timestamped file each
          invocation using :class:`logging.FileHandler`.
        - ``"single"``: uses a persistent file with
          :class:`~logging.handlers.RotatingFileHandler` for automatic
          rotation when the file exceeds ``rotation_max_bytes``.
        """

        # Resolve directory path
        dir_str = config.get("file_directory", DEFAULT_LOGGING_CONFIG["file_directory"])
        log_dir = Path(os.path.expanduser(dir_str))

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # Can't create directory, skip file logging
            sys.stderr.write(f"Warning: Cannot create log directory {log_dir}: {e}\n")
            return None, None

        file_mode = config.get("file_mode", "per_run")

        if file_mode == "single":
            # --- Single persistent file with rotation ---
            name_pattern = config.get("file_single_name", "{app}.log")
            try:
                filename = name_pattern.format(app=app_name)
            except (KeyError, ValueError):
                filename = f"{app_name}.log"

            log_file_path = log_dir / filename
            max_bytes = config.get("rotation_max_bytes", 10 * 1024 * 1024)
            backup_count = config.get("rotation_backup_count", 5)

            try:
                handler = RotatingFileHandler(
                    log_file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            except (OSError, PermissionError) as e:
                sys.stderr.write(f"Warning: Cannot create log file {log_file_path}: {e}\n")
                return None, None
        else:
            # --- Per-run file (current behavior, unchanged) ---
            pattern = config.get("file_name_pattern", DEFAULT_LOGGING_CONFIG["file_name_pattern"])
            timestamp = datetime.now()

            try:
                filename = pattern.format(app=app_name, timestamp=timestamp)
            except (KeyError, ValueError):
                filename = f"{app_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.log"

            log_file_path = log_dir / filename

            try:
                handler = logging.FileHandler(log_file_path, encoding="utf-8")
            except (OSError, PermissionError) as e:
                sys.stderr.write(f"Warning: Cannot create log file {log_file_path}: {e}\n")
                return None, None

        # Level and format - same for both modes
        level_str = config.get("file_level", "DEBUG").upper()
        level = logging.getLevelName(level_str)
        handler.setLevel(level if isinstance(level, int) else logging.DEBUG)

        fmt = config.get("file_format", DEFAULT_LOGGING_CONFIG["file_format"])
        handler.setFormatter(logging.Formatter(fmt))

        return handler, log_file_path

    def set_console_level(self, level: str | int) -> None:
        """Change the console handler's log level at runtime."""
        if self._console_handler is None:
            return

        if isinstance(level, str):
            level = logging.getLevelName(level.upper())

        if isinstance(level, int):
            self._console_handler.setLevel(level)

    def set_file_level(self, level: str | int) -> None:
        """Change the file handler's log level at runtime."""
        if self._file_handler is None:
            return

        if isinstance(level, str):
            level = logging.getLevelName(level.upper())

        if isinstance(level, int):
            self._file_handler.setLevel(level)

    def set_component_level(self, component: str, level: str | int) -> None:
        """Change a specific component's log level at runtime."""
        logger = logging.getLogger(component)

        if isinstance(level, str):
            level = logging.getLevelName(level.upper())

        if isinstance(level, int):
            logger.setLevel(level)

    def disable_console(self) -> None:
        """Disable console logging entirely.

        After this call, even ``display=True`` records will not appear
        on the console until :meth:`enable_console` is called.
        """
        if self._console_handler:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._console_handler)
            self._console_handler = None
            self._display_filter = None

    def enable_console(self, level: str = "WARNING") -> None:
        """Re-enable console logging.

        Creates a new console handler with a ``DisplayFilter`` that
        passes **all** records (console globally enabled).  If a handler
        already exists (e.g. from ``configure()`` with
        ``console_enabled=False``), it is replaced so the filter is
        upgraded to globally enabled.
        """
        # If already fully enabled, nothing to do.
        if self._display_filter is not None and self._display_filter.console_globally_enabled:
            return

        # Remove existing handler if present (it may have a restrictive filter).
        if self._console_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._console_handler)

        config = {
            "console_level": level,
            "console_format": DEFAULT_LOGGING_CONFIG["console_format"],
        }
        self._console_handler = self._create_console_handler(config)
        # When explicitly enabling, set console_globally_enabled=True
        # so all records pass through (filter defers to handler level).
        self._display_filter = DisplayFilter(
            console_globally_enabled=True,
            display_min_level=logging.DEBUG,
        )
        self._console_handler.addFilter(self._display_filter)
        logging.getLogger().addHandler(self._console_handler)


# ---------------------------------------------------------------------------
# Public module-level functions
# ---------------------------------------------------------------------------


def configure_logging(
    app_name: str = "llmcore",
    config: dict[str, Any] | None = None,
    config_file_path: str | Path | None = None,
    force_reconfigure: bool = False,
) -> Path:
    """
    Configure unified logging for the application.

    This is the main entry point for configuring logging. Call this
    early in your application startup, before importing modules that
    use logging.

    Args:
        app_name: Name of the application (used in log filename)
        config: Pre-loaded logging configuration dictionary
        config_file_path: Path to config file (if config not provided)
        force_reconfigure: If True, reconfigure even if already configured

    Returns:
        Path to the log file (if file logging enabled)

    Example:
        # Simple usage (reads from default config locations):
        from llmcore.logging_config import configure_logging
        configure_logging(app_name="llmchat")

        # With explicit config:
        configure_logging(
            app_name="llmchat",
            config={
                "console_enabled": False,
                "file_level": "DEBUG",
                "file_directory": "/var/log/myapp"
            }
        )
    """
    manager = UnifiedLoggingManager.get_instance()
    return manager.configure(
        app_name=app_name,
        config=config,
        config_file_path=config_file_path,
        force_reconfigure=force_reconfigure,
    )


def log_display(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Log a message that also appears on console even in silent mode.

    Convenience wrapper around ``logger.log()`` that sets
    ``extra={"display": True}``.  The message goes to the file handler
    (as always) **and** passes through the :class:`DisplayFilter` to the
    console handler.

    The ``display_min_level`` setting still applies - a DEBUG record with
    ``display=True`` won't show if ``display_min_level`` is INFO.

    Args:
        logger: Logger instance to use.
        level: Logging level (e.g., ``logging.INFO``, ``logging.WARNING``).
        msg: Message format string.
        *args: Format arguments.
        **kwargs: Passed through to ``logger.log()``.  The ``extra``
            kwarg is merged (not replaced) to preserve caller's extra
            data.

    Example::

        from llmcore.logging_config import log_display
        import logging

        logger = logging.getLogger("semantiscan.indexing")
        log_display(logger, logging.INFO, "Indexing %d files in %s", count, path)
        log_display(logger, logging.WARNING, "Slow query: %.2fs", elapsed)
    """
    extra = kwargs.pop("extra", None) or {}
    extra["display"] = True
    kwargs["extra"] = extra
    logger.log(level, msg, *args, **kwargs)


def get_log_file_path() -> Path | None:
    """Get the current log file path."""
    return UnifiedLoggingManager.get_log_file_path()


def set_console_level(level: str | int) -> None:
    """Change console log level at runtime."""
    UnifiedLoggingManager.get_instance().set_console_level(level)


def set_file_level(level: str | int) -> None:
    """Change file log level at runtime."""
    UnifiedLoggingManager.get_instance().set_file_level(level)


def set_component_level(component: str, level: str | int) -> None:
    """Change a specific component's log level at runtime."""
    UnifiedLoggingManager.get_instance().set_component_level(component, level)


def disable_console_logging() -> None:
    """Disable all console logging."""
    UnifiedLoggingManager.get_instance().disable_console()


def enable_console_logging(level: str = "WARNING") -> None:
    """Re-enable console logging."""
    UnifiedLoggingManager.get_instance().enable_console(level)
