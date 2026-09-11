"""Validate special-output schemas before any comparison-time coercion."""

from __future__ import annotations

import numpy as np


class OutputContractError(ValueError):
    """Raised when a verifier output violates its declared wire/schema contract."""


def require_array(
    value,
    *,
    name: str,
    dtype,
    shape=None,
    finite: bool = False,
):
    """Return ``value`` as an array after exact dtype/shape validation."""
    array = np.asarray(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise OutputContractError(
            f"{name} dtype must be {expected_dtype}, got {array.dtype}"
        )
    if shape is not None and array.shape != tuple(shape):
        raise OutputContractError(
            f"{name} shape must be {tuple(shape)}, got {array.shape}"
        )
    if finite and not np.all(np.isfinite(array)):
        raise OutputContractError(f"{name} must contain only finite values")
    return array


def require_shape(value, *, name: str, shape):
    """Validate shape without changing or constraining the storage dtype."""
    array = np.asarray(value)
    if array.shape != tuple(shape):
        raise OutputContractError(
            f"{name} shape must be {tuple(shape)}, got {array.shape}"
        )
    return array


def require_packed_uint8_field(value, *, name: str):
    """Validate a float32 wire field that semantically carries uint8 values."""
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise OutputContractError(f"{name} must contain only finite values")
    if not np.all(array == np.rint(array)):
        raise OutputContractError(f"{name} must contain integral values")
    if np.any(array < 0) or np.any(array > 255):
        raise OutputContractError(f"{name} values must be in [0, 255]")
    return array


def require_binary_uint8(value, *, name: str, shape):
    """Validate a uint8 wire field whose logical values are booleans."""
    array = require_array(value, name=name, dtype=np.uint8, shape=shape)
    if np.any((array != 0) & (array != 1)):
        raise OutputContractError(f"{name} must contain only 0 or 1")
    return array
