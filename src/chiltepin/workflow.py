# SPDX-License-Identifier: Apache-2.0

"""Workflow context managers for Chiltepin.

This module provides context managers that wrap Parsl configuration and lifecycle
management, eliminating the need for users to directly import or interact with Parsl.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import parsl
from globus_compute_sdk import Client

from chiltepin import configure

# Module-level logger for cleanup warnings
_logger = logging.getLogger(__name__)


class Workflow:
    """Workflow context manager for Chiltepin.

    Can be used as a context manager or with explicit start/cleanup calls.
    This wraps Parsl configuration and lifecycle management, eliminating the need
    for users to directly import or interact with Parsl.

    Parameters
    ----------
    config : str, Path, or dict
        Either a path to a YAML configuration file or a configuration dictionary
    include : list of str, optional
        List of resource labels to load. If None, all resources are loaded.
        Note: The default "local" resource is always available, regardless of
        this parameter.
    run_dir : str, optional
        Directory for Parsl runtime files. If None, uses Parsl's default.
    client : globus_compute_sdk.Client, optional
        Globus Compute client for Globus Compute resources. If None, one will
        be created automatically if needed.
    log_file : str, optional
        Path to Parsl log file. If None, no file logging is configured.
    log_level : int, optional
        Logging level (e.g., logging.DEBUG). Only used if log_file is provided.

    Examples
    --------
    As a context manager:

    >>> from chiltepin import Workflow
    >>> from chiltepin.tasks import python_task
    >>>
    >>> @python_task
    >>> def my_task():
    ...     return "Hello!"
    >>>
    >>> with Workflow("config.yaml") as dfk:
    ...     result = my_task()
    ...     print(result.result())

    With explicit start/cleanup (for use across methods):

    >>> workflow = Workflow("config.yaml")
    >>> dfk = workflow.start()
    >>> # ... do work, potentially in other methods ...
    >>> workflow.cleanup()
    """

    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        *,
        include: Optional[List[str]] = None,
        run_dir: Optional[str] = None,
        client: Optional[Client] = None,
        log_file: Optional[str] = None,
        log_level: Optional[int] = None,
    ):
        """Initialize workflow configuration (does not start workflow yet)."""
        # Parse config
        if isinstance(config, (str, Path)):
            self.config_dict = configure.parse_file(str(config))
        else:
            self.config_dict = config

        self.include = include
        self.run_dir = run_dir
        self.client = client
        self.log_file = log_file
        self.log_level = log_level
        self.dfk = None
        self.logger_handler = None

    def start(self):
        """Start the workflow and return DataFlowKernel.

        Returns
        -------
        DataFlowKernel
            The Parsl DataFlowKernel instance. Can be used to access workflow
            state (e.g., dfk.tasks) or for advanced operations.

        Raises
        ------
        RuntimeError
            If workflow is already started.
        """
        if self.dfk is not None:
            raise RuntimeError("Workflow already started. Call cleanup() first.")

        try:
            # Set up logging if requested
            if self.log_file is not None:
                import logging as log_module

                level = (
                    self.log_level if self.log_level is not None else log_module.INFO
                )
                self.logger_handler = parsl.set_file_logger(
                    filename=self.log_file, level=level
                )

            # Load configuration
            parsl_config = configure.load(
                self.config_dict,
                include=self.include,
                client=self.client,
                run_dir=self.run_dir,
            )

            # Load Parsl
            self.dfk = parsl.load(parsl_config)
            return self.dfk
        except Exception:
            # Best-effort cleanup of any partial initialization
            # This ensures explicit start()/cleanup() usage is as safe as context manager
            self.cleanup(suppress_exceptions=True)
            raise

    def cleanup(self, suppress_exceptions=False):
        """Cleanup the workflow and release resources.

        This should be called when the workflow is complete. If using as a
        context manager, this is called automatically.

        Parameters
        ----------
        suppress_exceptions : bool
            If True, exceptions during cleanup are logged but not raised.
            This is used when cleaning up after a user exception to ensure
            the user's exception is not masked by cleanup exceptions.

        Notes
        -----
        All cleanup operations are attempted even if some fail. If multiple
        cleanup operations raise exceptions and suppress_exceptions is False,
        they are chained together using __cause__, with the last exception
        being raised.
        """
        cleanup_exception = None

        # Attempt dfk.cleanup()
        if self.dfk is not None:
            try:
                self.dfk.cleanup()
            except Exception as e:
                if suppress_exceptions:
                    _logger.warning(
                        "Exception during dfk.cleanup() (suppressing)",
                        exc_info=True,
                    )
                else:
                    _logger.warning("Exception during dfk.cleanup()", exc_info=True)
                    cleanup_exception = e

        # Always attempt parsl.clear()
        try:
            parsl.clear()
        except Exception as e:
            if suppress_exceptions:
                _logger.warning(
                    "Exception during parsl.clear() (suppressing)",
                    exc_info=True,
                )
            else:
                _logger.warning("Exception during parsl.clear()", exc_info=True)
                if cleanup_exception is None:
                    cleanup_exception = e
                else:
                    # Chain this exception to the previous one
                    e.__cause__ = cleanup_exception
                    cleanup_exception = e

        # Always attempt logger cleanup
        if self.logger_handler is not None:
            try:
                self.logger_handler()
            except Exception as e:
                if suppress_exceptions:
                    _logger.warning(
                        "Exception during logger cleanup (suppressing)",
                        exc_info=True,
                    )
                else:
                    _logger.warning("Exception during logger cleanup", exc_info=True)
                    if cleanup_exception is None:
                        cleanup_exception = e
                    else:
                        # Chain this exception to the previous one
                        e.__cause__ = cleanup_exception
                        cleanup_exception = e

        # Only reset state if cleanup succeeded
        # If cleanup failed, preserve state for debugging and potential retry
        if cleanup_exception is None or suppress_exceptions:
            self.dfk = None
            self.logger_handler = None

        # Raise the final exception if any occurred and not suppressing
        if cleanup_exception is not None and not suppress_exceptions:
            raise cleanup_exception

    def __enter__(self):
        """Context manager entry - starts the workflow.

        If start() fails, it performs its own cleanup before re-raising.
        """
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up the workflow."""
        # Suppress cleanup exceptions if user exception occurred
        user_exception = exc_type is not None
        self.cleanup(suppress_exceptions=user_exception)
        return False  # Don't suppress user exceptions


__all__ = [
    "Workflow",
]
