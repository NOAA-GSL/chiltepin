# SPDX-License-Identifier: Apache-2.0

"""Chiltepin: Federated NWP Workflow Tools.

This package provides tools for building scientific workflows that can execute
on distributed computing resources using Parsl and Globus services.
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("chiltepin")
except PackageNotFoundError:
    # Fallback for development installs or when package is not installed
    __version__ = "dev"

__all__ = [
    "Workflow",
]


def __getattr__(name):
    """Lazy import of Workflow to avoid loading Parsl unnecessarily.

    The Workflow class is not imported until explicitly accessed, avoiding the
    overhead of loading Parsl and its dependencies when they're not needed.
    This also enables attribute-style access to submodules (e.g., chiltepin.configure).
    """
    if name in __all__:
        from chiltepin.workflow import Workflow  # noqa: F401

        globals()[name] = locals()[name]
        return locals()[name]

    # Try to import as a submodule
    try:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    except ModuleNotFoundError as e:
        # Only suppress if the error is for the module we're trying to import,
        # not for some dependency that module is trying to import
        if e.name != f"{__name__}.{name}":
            raise  # Some other module wasn't found, propagate the error

    # If not a submodule, raise AttributeError as normal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
