# /**
#   ******************************************************************************
#   * @file        runner_params.py
#   * @author      Egor Izmaylov
#   * @brief       Shared parameter packing helpers for numerical runner plans.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import numpy as np


_RECURRENT_MAX_ACTIVATIONS = 6
_RECURRENT_ACTIVATION_CODES = {
    "tanh": 0,
    "sigmoid": 1,
    "relu": 2,
    "affine": 3,
    "leakyrelu": 4,
    "thresholdedrelu": 5,
    "scaledtanh": 6,
    "hardsigmoid": 7,
    "elu": 8,
    "softsign": 9,
    "softplus": 10,
}


def recurrent_params_binary(base_values, init_args):
    activations = init_args.get("activations", []) or []
    alphas = init_args.get("activation_alpha", []) or []
    betas = init_args.get("activation_beta", []) or []
    codes = [0] * _RECURRENT_MAX_ACTIVATIONS
    alpha_values = [np.nan] * _RECURRENT_MAX_ACTIVATIONS
    beta_values = [np.nan] * _RECURRENT_MAX_ACTIVATIONS
    for idx, activation in enumerate(list(activations)[:_RECURRENT_MAX_ACTIVATIONS]):
        name = activation.decode("utf-8") if isinstance(activation, bytes) else activation
        key = str(name).lower()
        if key not in _RECURRENT_ACTIVATION_CODES:
            raise ValueError(f"Unsupported recurrent activation {activation!r}")
        codes[idx] = _RECURRENT_ACTIVATION_CODES[key]
    for idx, value in enumerate(list(alphas)[:_RECURRENT_MAX_ACTIVATIONS]):
        alpha_values[idx] = float(value)
    for idx, value in enumerate(list(betas)[:_RECURRENT_MAX_ACTIVATIONS]):
        beta_values[idx] = float(value)
    clip = init_args.get("clip", None)
    int_values = list(base_values) + [min(len(activations), _RECURRENT_MAX_ACTIVATIONS), int(clip is not None)] + codes
    float_values = alpha_values + beta_values + [0.0 if clip is None else float(clip)]
    return np.array(int_values, dtype=np.int32).tobytes() + np.array(float_values, dtype=np.float32).tobytes()


def slice_io_values(init_args, input_shape):
    rank = len(input_shape)
    starts = np.asarray(init_args.get("starts_value", [0] * rank), dtype=np.int64).reshape(-1)
    ends = np.asarray(init_args.get("ends_value", list(input_shape)), dtype=np.int64).reshape(-1)
    axes = np.asarray(init_args.get("axes_value", list(range(len(starts)))), dtype=np.int64).reshape(-1)
    steps = np.asarray(init_args.get("steps_value", [1] * len(starts)), dtype=np.int64).reshape(-1)
    return starts, ends, axes, steps


def normalize_slice_parameters(input_shape, starts, ends, axes, steps):
    rank = len(input_shape)
    full_starts = [0] * rank
    full_ends = list(map(int, input_shape))
    full_steps = [1] * rank

    for idx, axis in enumerate(axes.tolist()):
        if axis < 0:
            axis += rank
        step = int(steps[idx])
        if step == 0:
            raise ValueError("Slice step must not be zero")
        start = int(starts[idx])
        end = int(ends[idx])
        dim_len = int(input_shape[axis])
        if start < 0:
            start += dim_len
        if end < 0:
            end += dim_len
        if step > 0:
            start = max(0, min(start, dim_len))
            end = max(0, min(end, dim_len))
        else:
            start = max(0, min(start, dim_len - 1))
            end = max(-1, min(end, dim_len - 1))
        full_starts[axis] = start
        full_ends[axis] = end
        full_steps[axis] = step

    return full_starts, full_ends, full_steps


def onnx_dtype_id_from_name(dtype_name):
    mapping = {
        "float32": 1,
        "uint8": 2,
        "int8": 3,
        "uint16": 4,
        "int16": 5,
        "int32": 6,
        "int64": 7,
        "bool": 9,
        "float16": 10,
        "float64": 11,
        "uint32": 12,
        "uint64": 13,
        "bfloat16": 16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported ONNX output_datatype for numerical window plan: {dtype_name}")
    return mapping[dtype_name]
