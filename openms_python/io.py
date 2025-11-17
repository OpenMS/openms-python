"""Utilities for reading and writing mzML files with pyOpenMS."""
from __future__ import annotations

from pathlib import Path
from typing import Union

try:
    import pyopenms as oms
except ImportError:  # pragma: no cover - handled gracefully at runtime
    oms = None  # type: ignore


class _PyOpenMSNotAvailable(RuntimeError):
    """Raised when pyOpenMS is required but not installed."""


def _ensure_pyopenms_available() -> None:
    """Ensure pyOpenMS is importable before performing IO."""

    if oms is None:
        raise _PyOpenMSNotAvailable(
            "pyOpenMS is required for reading/writing mzML files. "
            "Install the 'pyopenms' package to use these helpers."
        )


def read_mzml(path: Union[str, Path], *, as_wrapper: bool = True):
    """Load an mzML file and optionally wrap it in :class:`Py_MSExperiment`."""

    from .py_msexperiment import Py_MSExperiment  # Local import to avoid cycles

    _ensure_pyopenms_available()
    mzml = oms.MzMLFile()
    experiment = oms.MSExperiment()
    mzml.load(str(path), experiment)
    return Py_MSExperiment(experiment) if as_wrapper else experiment


def write_mzml(experiment, path: Union[str, Path]) -> None:
    """Persist an MSExperiment or Py_MSExperiment to disk."""

    _ensure_pyopenms_available()
    native = getattr(experiment, "_experiment", experiment)
    if not isinstance(native, oms.MSExperiment):
        raise TypeError(
            "write_mzml expects a pyopenms.MSExperiment or Py_MSExperiment instance"
        )

    mzml = oms.MzMLFile()
    mzml.store(str(path), native)
