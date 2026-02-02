# src/llmcore/logging_config.py
"""
Unified Logging Configuration for the LLMCore Ecosystem.

This module provides a centralized logging setup that can be used by all
components (llmcore, llmchat, semantiscan, confy) to ensure consistent
logging behavior across the entire application stack.

Configuration is read from confy (the single source of truth) and supports:
- Console logging (can be disabled entirely)
- File logging with configurable directory and filename patterns
- Per-component log level overrides
- Consistent formatting across all components

Usage:
    from llmcore.logging_config import configure_logging

    # Early in application startup (before other imports that use logging):
    configure_logging(app_name="llmchat")

    # Or with a pre-loaded config:
    configure_logging(app_name="llmchat", config=my_confy_config)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Try to import confy for config loading
try:
    from confy import Confy

    CONFY_AVAILABLE = True
except ImportError:
    CONFY_AVAILABLE = False


# Default logging configuration (used if no config file found)
DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "console_enabled": True,
    "console_level": "WARNING",
    "console_format": "%(levelname)s - %(message)s",
    "file_enabled": True,
    "file_level": "DEBUG",
    "file_directory": "~/.local/share/llmcore/logs",
    "file_name_pattern": "{app}_{timestamp:%Y%m%d_%H%M%S}.log",
    "file_format": "%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)",
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


class UnifiedLoggingManager:
    """
    Singleton manager for unified logging configuration.

    Ensures logging is only configured once and provides methods
    for runtime adjustment of log levels.
    """

    _instance: Optional["UnifiedLoggingManager"] = None
    _configured: bool = False
    _log_file_path: Optional[Path] = None
    _console_handler: Optional[logging.Handler] = None
    _file_handler: Optional[logging.Handler] = None

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
    def get_log_file_path(cls) -> Optional[Path]:
        """Get the current log file path."""
        return cls._log_file_path

    def configure(
        self,
        app_name: str = "llmcore",
        config: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[Union[str, Path]] = None,
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

        # Configure console handler
        if log_config.get("console_enabled", True):
            self._console_handler = self._create_console_handler(log_config)
            root_logger.addHandler(self._console_handler)

        # Configure file handler
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
        self, config: Optional[Dict[str, Any]], config_file_path: Optional[Union[str, Path]]
    ) -> Dict[str, Any]:
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

    def _create_console_handler(self, config: Dict[str, Any]) -> logging.Handler:
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
        self, config: Dict[str, Any], app_name: str
    ) -> tuple[Optional[logging.Handler], Optional[Path]]:
        """Create and configure the file handler."""

        # Resolve directory path
        dir_str = config.get("file_directory", DEFAULT_LOGGING_CONFIG["file_directory"])
        log_dir = Path(os.path.expanduser(dir_str))

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # Can't create directory, skip file logging
            sys.stderr.write(f"Warning: Cannot create log directory {log_dir}: {e}\n")
            return None, None

        # Generate filename from pattern
        pattern = config.get("file_name_pattern", DEFAULT_LOGGING_CONFIG["file_name_pattern"])
        timestamp = datetime.now()

        try:
            filename = pattern.format(app=app_name, timestamp=timestamp)
        except (KeyError, ValueError):
            filename = f"{app_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.log"

        log_file_path = log_dir / filename

        # Create handler
        try:
            handler = logging.FileHandler(log_file_path, encoding="utf-8")
        except (OSError, PermissionError) as e:
            sys.stderr.write(f"Warning: Cannot create log file {log_file_path}: {e}\n")
            return None, None

        # Set level
        level_str = config.get("file_level", "DEBUG").upper()
        level = logging.getLevelName(level_str)
        if isinstance(level, int):
            handler.setLevel(level)
        else:
            handler.setLevel(logging.DEBUG)

        # Set format
        fmt = config.get("file_format", DEFAULT_LOGGING_CONFIG["file_format"])
        handler.setFormatter(logging.Formatter(fmt))

        return handler, log_file_path

    def set_console_level(self, level: Union[str, int]) -> None:
        """Change the console handler's log level at runtime."""
        if self._console_handler is None:
            return

        if isinstance(level, str):
            level = logging.getLevelName(level.upper())

        if isinstance(level, int):
            self._console_handler.setLevel(level)

    def set_file_level(self, level: Union[str, int]) -> None:
        """Change the file handler's log level at runtime."""
        if self._file_handler is None:
            return

        if isinstance(level, str):
            level = logging.getLevelName(level.upper())

        if isinstance(level, int):
            self._file_handler.setLevel(level)

    def set_component_level(self, component: str, level: Union[str, int]) -> None:
        """Change a specific component's log level at runtime."""
        logger = logging.getLogger(component)

        if isinstance(level, str):
            level = logging.getLevelName(level.upper())

        if isinstance(level, int):
            logger.setLevel(level)

    def disable_console(self) -> None:
        """Disable console logging entirely."""
        if self._console_handler:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._console_handler)
            self._console_handler = None

    def enable_console(self, level: str = "WARNING") -> None:
        """Re-enable console logging."""
        if self._console_handler is not None:
            return  # Already enabled

        config = {
            "console_level": level,
            "console_format": DEFAULT_LOGGING_CONFIG["console_format"],
        }
        self._console_handler = self._create_console_handler(config)
        logging.getLogger().addHandler(self._console_handler)


def configure_logging(
    app_name: str = "llmcore",
    config: Optional[Dict[str, Any]] = None,
    config_file_path: Optional[Union[str, Path]] = None,
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


def get_log_file_path() -> Optional[Path]:
    """Get the current log file path."""
    return UnifiedLoggingManager.get_log_file_path()


def set_console_level(level: Union[str, int]) -> None:
    """Change console log level at runtime."""
    UnifiedLoggingManager.get_instance().set_console_level(level)


def set_file_level(level: Union[str, int]) -> None:
    """Change file log level at runtime."""
    UnifiedLoggingManager.get_instance().set_file_level(level)


def set_component_level(component: str, level: Union[str, int]) -> None:
    """Change a specific component's log level at runtime."""
    UnifiedLoggingManager.get_instance().set_component_level(component, level)


def disable_console_logging() -> None:
    """Disable all console logging."""
    UnifiedLoggingManager.get_instance().disable_console()


def enable_console_logging(level: str = "WARNING") -> None:
    """Re-enable console logging."""
    UnifiedLoggingManager.get_instance().enable_console(level)
