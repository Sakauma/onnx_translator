# /**
#   ******************************************************************************
#   * @file        common.py
#   * @author      Egor Izmaylov
#   * @brief       提供算子实现模块共享的导入、ctypes 参数结构和辅助函数。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from nn import Ops
from nn import Tensor, Tensor_, DTYPE_MAP, CTensor
import nn
import ctypes
import numpy as np
from typing import List, Union
import os
import unicodedata


def _float32_to_bfloat16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


def _bfloat16_bits_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def _tensor_data_as_numeric(tensor):
    if getattr(tensor, "dtype", None) == "bfloat16":
        return _bfloat16_bits_to_float32(tensor.data)
    return np.asarray(tensor.data)


def _cast_numeric_to_dtype(values, dtype):
    if dtype == "bfloat16":
        return _float32_to_bfloat16_bits(values)
    return np.asarray(values, dtype=nn.DTYPE_TO_NUMPY.get(dtype, np.asarray(values).dtype))


def _conv_attr(values, spatial_rank, default):
    if values is None:
        return [default] * spatial_rank
    values = list(values)
    if len(values) != spatial_rank:
        raise ValueError(f"Convolution attribute rank {len(values)} does not match spatial rank {spatial_rank}")
    return values

def _conv_effective_kernel(kernel_shape, dilations):
    return [dilations[i] * (kernel_shape[i] - 1) + 1 for i in range(len(kernel_shape))]

def _conv_resolve_pads(input_spatial, kernel_shape, pads, strides, dilations, auto_pad="NOTSET"):
    spatial_rank = len(input_spatial)
    auto_pad = auto_pad or "NOTSET"
    if auto_pad == "VALID":
        return [0] * (2 * spatial_rank)
    if auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
        effective = _conv_effective_kernel(kernel_shape, dilations)
        resolved = []
        end_pads = []
        for dim in range(spatial_rank):
            out_dim = int(np.ceil(float(input_spatial[dim]) / float(strides[dim])))
            total = max((out_dim - 1) * strides[dim] + effective[dim] - input_spatial[dim], 0)
            if auto_pad == "SAME_UPPER":
                begin = total // 2
            else:
                begin = total - total // 2
            resolved.append(begin)
            end_pads.append(total - begin)
        return resolved + end_pads
    if pads is None:
        return [0] * (2 * spatial_rank)
    pads = list(pads)
    if len(pads) != 2 * spatial_rank:
        raise ValueError(f"Convolution pads must contain {2 * spatial_rank} values")
    return pads

def _conv_output_spatial(input_spatial, kernel_shape, pads, strides, dilations):
    effective = _conv_effective_kernel(kernel_shape, dilations)
    spatial_rank = len(input_spatial)
    return tuple(
        (input_spatial[i] + pads[i] + pads[spatial_rank + i] - effective[i]) // strides[i] + 1
        for i in range(spatial_rank)
    )

def _conv_nd_numpy(x, w, bias=None, pads=None, strides=None, dilations=None, group=1, auto_pad="NOTSET", acc_dtype=np.float64):
    x = np.asarray(x)
    w = np.asarray(w)
    spatial_rank = x.ndim - 2
    kernel_shape = list(w.shape[2:])
    strides = _conv_attr(strides, spatial_rank, 1)
    dilations = _conv_attr(dilations, spatial_rank, 1)
    pads = _conv_resolve_pads(list(x.shape[2:]), kernel_shape, pads, strides, dilations, auto_pad)

    n_batches, in_channels = x.shape[:2]
    out_channels, channels_per_group = w.shape[:2]
    if group <= 0 or in_channels % group != 0 or out_channels % group != 0:
        raise ValueError(f"Invalid convolution group={group} for input channels={in_channels}, output channels={out_channels}")
    if channels_per_group != in_channels // group:
        raise ValueError(f"Weight channel dimension {channels_per_group} does not match input channels/group {in_channels // group}")

    out_spatial = _conv_output_spatial(list(x.shape[2:]), kernel_shape, pads, strides, dilations)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[spatial_rank + i]) for i in range(spatial_rank)]
    x_pad = np.pad(x.astype(acc_dtype, copy=False), pad_width, mode="constant")
    w_acc = w.astype(acc_dtype, copy=False)
    out = np.zeros((n_batches, out_channels) + out_spatial, dtype=acc_dtype)
    out_per_group = out_channels // group
    in_per_group = in_channels // group

    for n in range(n_batches):
        for g in range(group):
            for oc_local in range(out_per_group):
                oc = g * out_per_group + oc_local
                for out_index in np.ndindex(*out_spatial):
                    total = 0
                    for ic_local in range(in_per_group):
                        ic = g * in_per_group + ic_local
                        for kernel_index in np.ndindex(*kernel_shape):
                            in_index = tuple(out_index[d] * strides[d] + kernel_index[d] * dilations[d] for d in range(spatial_rank))
                            total += x_pad[(n, ic) + in_index] * w_acc[(oc, ic_local) + kernel_index]
                    out[(n, oc) + out_index] = total

    if bias is not None:
        bias_arr = np.asarray(bias, dtype=acc_dtype).reshape((1, out_channels) + (1,) * spatial_rank)
        out = out + bias_arr
    return out

def _reshape_channel_param(param, target, axis, dtype):
    if param is None:
        return np.array(0, dtype=dtype)
    arr = np.asarray(_tensor_data_as_numeric(param), dtype=dtype)
    if arr.ndim == 0 or arr.size == 1:
        return arr.reshape(())
    if arr.ndim == 1 and arr.shape[0] == target.shape[axis]:
        shape = [1] * target.ndim
        shape[axis] = arr.shape[0]
        return arr.reshape(shape)
    return arr

def _broadcast_conv_zero_point(param, target_shape, dtype, axis=None):
    np_dtype = nn.DTYPE_TO_NUMPY[dtype]
    if param is None:
        return np.zeros(target_shape, dtype=np_dtype)
    arr = np.asarray(param.data, dtype=np_dtype)
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    if arr.shape == () or arr.size == 1:
        return np.broadcast_to(arr.reshape(()), target_shape).copy()
    if axis is not None and arr.ndim == 1 and arr.shape[0] == target_shape[axis]:
        shape = [1] * len(target_shape)
        shape[axis] = arr.shape[0]
        return np.broadcast_to(arr.reshape(shape), target_shape).copy()
    return np.broadcast_to(arr, target_shape).copy()

def _broadcast_conv_param(param, target_shape, dtype, axis=None):
    np_dtype = nn.DTYPE_TO_NUMPY[dtype]
    arr = np.asarray(param.data, dtype=np_dtype)
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    if arr.shape == () or arr.size == 1:
        return np.broadcast_to(arr.reshape(()), target_shape).copy()
    if axis is not None and arr.ndim == 1 and arr.shape[0] == target_shape[axis]:
        shape = [1] * len(target_shape)
        shape[axis] = arr.shape[0]
        return np.broadcast_to(arr.reshape(shape), target_shape).copy()
    return np.broadcast_to(arr, target_shape).copy()

def _reshape_output_channel_param(param, out_channels, spatial_rank, dtype):
    arr = np.asarray(_tensor_data_as_numeric(param), dtype=dtype)
    if arr.ndim == 0 or arr.size == 1:
        return arr.reshape(())
    if arr.ndim == 1 and arr.shape[0] == out_channels:
        return arr.reshape((1, out_channels) + (1,) * spatial_rank)
    return arr

def _dtype_bounds(dtype):
    np_dtype = nn.DTYPE_TO_NUMPY.get(dtype, np.uint8)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        return info.min, info.max
    return None, None

def _normalize_pool_params(input_shape, kernel_shape, pads, strides, dilations, auto_pad="NOTSET"):
    spatial_rank = len(input_shape) - 2
    if spatial_rank < 1:
        raise ValueError("Pool operators expect input rank >= 3")
    if len(kernel_shape) != spatial_rank:
        raise ValueError(f"kernel_shape rank {len(kernel_shape)} does not match input spatial rank {spatial_rank}")

    strides = list(strides) if strides else [1] * spatial_rank
    dilations = list(dilations) if dilations else [1] * spatial_rank
    pads = _conv_resolve_pads(list(input_shape[2:]), list(kernel_shape), pads, strides, dilations, auto_pad)
    if len(pads) != 2 * spatial_rank:
        raise ValueError(f"pads must contain {2 * spatial_rank} values for spatial rank {spatial_rank}")
    if len(strides) != spatial_rank:
        raise ValueError(f"strides must contain {spatial_rank} values")
    if len(dilations) != spatial_rank:
        raise ValueError(f"dilations must contain {spatial_rank} values")
    return spatial_rank, pads, strides, dilations

def _pool_output_shape(input_shape, kernel_shape, pads, strides, dilations, ceil_mode=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(input_shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_spatial = []
    for axis in range(spatial_rank):
        input_dim = input_shape[axis + 2]
        kernel_extent = dilations[axis] * (kernel_shape[axis] - 1) + 1
        numerator = input_dim + pads[axis] + pads[axis + spatial_rank] - kernel_extent
        if ceil_mode:
            out_dim = int(np.ceil(numerator / strides[axis])) + 1
            if (out_dim - 1) * strides[axis] >= input_dim + pads[axis]:
                out_dim -= 1
        else:
            out_dim = numerator // strides[axis] + 1
        out_spatial.append(max(0, int(out_dim)))
    return tuple(input_shape[:2]) + tuple(out_spatial)

def _pool_window_slices(out_index, kernel_shape, pads, strides, dilations):
    spatial_rank = len(kernel_shape)
    slices = []
    for axis in range(spatial_rank):
        start = out_index[axis] * strides[axis]
        stop = start + dilations[axis] * (kernel_shape[axis] - 1) + 1
        slices.append(slice(start, stop, dilations[axis]))
    return tuple(slices)

def _pool_flat_index(input_shape, n, c, spatial_coords, storage_order=0):
    coords = (n, c, *spatial_coords)
    order = "F" if storage_order else "C"
    return int(np.ravel_multi_index(coords, input_shape, order=order))

def _max_pool_nd(data, kernel_shape, pads, strides, dilations, ceil_mode=0, storage_order=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(data.shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_shape = _pool_output_shape(data.shape, kernel_shape, pads, strides, dilations, ceil_mode)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
    work = np.pad(data, pad_width, mode="constant", constant_values=-np.inf)
    out = np.empty(out_shape, dtype=data.dtype)
    indices = np.zeros(out_shape, dtype=np.int64)

    for prefix in np.ndindex(data.shape[0], data.shape[1]):
        for out_spatial in np.ndindex(*out_shape[2:]):
            window = work[prefix + _pool_window_slices(out_spatial, kernel_shape, pads, strides, dilations)]
            flat = int(np.argmax(window))
            local = np.unravel_index(flat, window.shape)
            value = window[local]
            out[prefix + out_spatial] = value
            input_spatial = tuple(
                out_spatial[i] * strides[i] + local[i] * dilations[i] - pads[i]
                for i in range(spatial_rank)
            )
            if all(0 <= input_spatial[i] < data.shape[i + 2] for i in range(spatial_rank)):
                indices[prefix + out_spatial] = _pool_flat_index(data.shape, prefix[0], prefix[1], input_spatial, storage_order)
    return out, indices

def _average_pool_nd(data, kernel_shape, pads, strides, dilations, count_include_pad=0, ceil_mode=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(data.shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_shape = _pool_output_shape(data.shape, kernel_shape, pads, strides, dilations, ceil_mode)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
    work = np.pad(data, pad_width, mode="constant", constant_values=0)
    valid = np.pad(np.ones(data.shape, dtype=np.float32), pad_width, mode="constant", constant_values=0)
    out = np.empty(out_shape, dtype=np.float64)
    full_count = np.prod(kernel_shape)

    for prefix in np.ndindex(data.shape[0], data.shape[1]):
        for out_spatial in np.ndindex(*out_shape[2:]):
            slices = _pool_window_slices(out_spatial, kernel_shape, pads, strides, dilations)
            window = work[prefix + slices]
            if count_include_pad:
                denom = full_count
            else:
                denom = np.sum(valid[prefix + slices])
            out[prefix + out_spatial] = 0.0 if denom == 0 else np.sum(window) / denom
    return out

def _lp_pool_nd(data, kernel_shape, pads, strides, dilations, p=2, ceil_mode=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(data.shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_shape = _pool_output_shape(data.shape, kernel_shape, pads, strides, dilations, ceil_mode)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
    work = np.pad(data, pad_width, mode="constant", constant_values=0)
    out = np.empty(out_shape, dtype=np.float64)

    for prefix in np.ndindex(data.shape[0], data.shape[1]):
        for out_spatial in np.ndindex(*out_shape[2:]):
            window = work[prefix + _pool_window_slices(out_spatial, kernel_shape, pads, strides, dilations)]
            out[prefix + out_spatial] = np.sum(np.abs(window) ** p) ** (1.0 / p)
    return out

def _grid_denormalize(coord, length, align_corners):
    if align_corners:
        return (coord + 1.0) * (length - 1) / 2.0
    return ((coord + 1.0) * length - 1.0) / 2.0

def _reflect_coordinate(coord, low, high):
    if high <= low:
        return low
    span = high - low
    coord = abs((coord - low) % (2.0 * span))
    if coord > span:
        coord = 2.0 * span - coord
    return coord + low

def _sample_coordinate(coord, length, padding_mode, align_corners):
    if padding_mode == "border":
        return min(max(coord, 0.0), length - 1.0)
    if padding_mode == "reflection":
        low, high = (0.0, length - 1.0) if align_corners else (-0.5, length - 0.5)
        reflected = _reflect_coordinate(coord, low, high)
        return min(max(reflected, 0.0), length - 1.0)
    return coord

def _get_pixel_2d(data, y, x, padding_mode, align_corners):
    height, width = data.shape
    if padding_mode in {"border", "reflection"}:
        y = _sample_coordinate(y, height, padding_mode, align_corners)
        x = _sample_coordinate(x, width, padding_mode, align_corners)
    yi, xi = int(y), int(x)
    if yi < 0 or yi >= height or xi < 0 or xi >= width:
        return 0.0
    return data[yi, xi]

def _bilinear_sample_2d(data, y, x, padding_mode="zeros", align_corners=False):
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1
    ly = y - y0
    lx = x - x0
    hy = 1.0 - ly
    hx = 1.0 - lx
    return (
        _get_pixel_2d(data, y0, x0, padding_mode, align_corners) * hy * hx
        + _get_pixel_2d(data, y0, x1, padding_mode, align_corners) * hy * lx
        + _get_pixel_2d(data, y1, x0, padding_mode, align_corners) * ly * hx
        + _get_pixel_2d(data, y1, x1, padding_mode, align_corners) * ly * lx
    )

def _roi_align_weighted_terms(data, y, x):
    height, width = data.shape
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return (0.0, 0.0, 0.0, 0.0)
    y = max(y, 0.0)
    x = max(x, 0.0)
    y_low = int(y)
    x_low = int(x)
    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    return (
        data[y_low, x_low] * hy * hx,
        data[y_low, x_high] * hy * lx,
        data[y_high, x_low] * ly * hx,
        data[y_high, x_high] * ly * lx,
    )

def _cubic_coefficients(t):
    alpha = -0.75
    x = abs(t)
    return np.array([
        ((alpha * (x + 1) - 5 * alpha) * (x + 1) + 8 * alpha) * (x + 1) - 4 * alpha,
        ((alpha + 2) * x - (alpha + 3)) * x * x + 1,
        ((alpha + 2) * (1 - x) - (alpha + 3)) * (1 - x) * (1 - x) + 1,
        ((alpha * (2 - x) - 5 * alpha) * (2 - x) + 8 * alpha) * (2 - x) - 4 * alpha,
    ])

def _bicubic_sample_2d(data, y, x, padding_mode="zeros", align_corners=False):
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    cy = _cubic_coefficients(y - y0)
    cx = _cubic_coefficients(x - x0)
    total = 0.0
    for iy in range(4):
        for ix in range(4):
            total += cy[iy] * cx[ix] * _get_pixel_2d(data, y0 - 1 + iy, x0 - 1 + ix, padding_mode, align_corners)
    return total

def _num_directions(direction):
    if direction in ("forward", "reverse"):
        return 1
    if direction == "bidirectional":
        return 2
    raise ValueError(f"Unsupported recurrent direction {direction!r}")

def _recurrent_time_major(x, layout):
    return np.swapaxes(x, 0, 1) if layout == 1 else x

def _recurrent_output_layout(y, layout):
    return np.transpose(y, (2, 0, 1, 3)) if layout == 1 else y

def _activation_function(name, alpha=None, beta=None):
    name = name.decode("utf-8") if isinstance(name, bytes) else name
    if name in (None, "Tanh", "tanh"):
        return np.tanh
    if name in ("Sigmoid", "sigmoid"):
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if name in ("Relu", "relu"):
        return lambda x: np.maximum(x, 0)
    if name in ("Affine", "affine"):
        a = 1.0 if alpha is None else alpha
        b = 0.0 if beta is None else beta
        return lambda x: a * x + b
    if name in ("LeakyRelu", "leakyrelu"):
        a = 0.01 if alpha is None else alpha
        return lambda x: np.where(x >= 0, x, a * x)
    if name in ("ThresholdedRelu", "thresholdedrelu"):
        a = 1.0 if alpha is None else alpha
        return lambda x: np.where(x >= a, x, 0)
    if name in ("ScaledTanh", "scaledtanh"):
        a = 1.0 if alpha is None else alpha
        b = 1.0 if beta is None else beta
        return lambda x: a * np.tanh(b * x)
    if name in ("HardSigmoid", "hardsigmoid"):
        a = 0.2 if alpha is None else alpha
        b = 0.5 if beta is None else beta
        return lambda x: np.clip(a * x + b, 0, 1)
    if name in ("Elu", "elu"):
        a = 1.0 if alpha is None else alpha
        return lambda x: np.where(x >= 0, x, a * (np.exp(x) - 1))
    if name in ("Softsign", "softsign"):
        return lambda x: x / (1 + np.abs(x))
    if name in ("Softplus", "softplus"):
        return lambda x: np.log1p(np.exp(x))
    raise ValueError(f"Unsupported recurrent activation {name!r}")

def _activation_at(activations, alphas, betas, index, default):
    name = activations[index] if index < len(activations) else default
    alpha = alphas[index] if index < len(alphas) else None
    beta = betas[index] if index < len(betas) else None
    return _activation_function(name, alpha, beta)

_ACTIVATION_CODES = {
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

def _clip_if_needed(x, clip):
    return np.clip(x, -clip, clip) if clip is not None else x

def _sequence_mask(sequence_lens, t, batch_size):
    if sequence_lens is None:
        return np.ones((batch_size, 1), dtype=bool)
    return (np.asarray(sequence_lens.data).reshape(-1) > t).reshape(batch_size, 1)

def _matmul_output_shape(shape_a, shape_b):
    shape_a = list(shape_a)
    shape_b = list(shape_b)
    if len(shape_a) == 0 or len(shape_b) == 0:
        raise ValueError("MatMul inputs must have rank >= 1")
    is_a_1d = len(shape_a) == 1
    is_b_1d = len(shape_b) == 1
    if is_a_1d:
        shape_a = [1] + shape_a
    if is_b_1d:
        shape_b = shape_b + [1]
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"MatMul shape mismatch: {shape_a[-1]} != {shape_b[-2]}")
    batch = np.broadcast_shapes(tuple(shape_a[:-2]), tuple(shape_b[:-2]))
    out_shape = list(batch) + [shape_a[-2], shape_b[-1]]
    if is_b_1d:
        out_shape.pop(-1)
    if is_a_1d:
        out_shape.pop(-1 if is_b_1d else -2)
    return tuple(out_shape)

def _prepare_matmul_c_shapes(input_a: Tensor, input_b: Tensor):
    data_a = np.asarray(input_a.data)
    data_b = np.asarray(input_b.data)
    is_a_1d = data_a.ndim == 1
    is_b_1d = data_b.ndim == 1

    if is_a_1d:
        data_a = data_a[np.newaxis, :]
    if is_b_1d:
        data_b = data_b[:, np.newaxis]

    shape_a = list(data_a.shape)
    shape_b = list(data_b.shape)
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"MatMul shape mismatch: {shape_a[-1]} != {shape_b[-2]}")
    batch_out = np.broadcast_shapes(tuple(shape_a[:-2]), tuple(shape_b[:-2]))
    out_shape_for_c = tuple(list(batch_out) + [shape_a[-2], shape_b[-1]])

    final_shape = list(out_shape_for_c)
    if is_b_1d:
        final_shape.pop(-1)
    if is_a_1d:
        final_shape.pop(-1 if is_b_1d else -2)
    return data_a, data_b, out_shape_for_c, tuple(final_shape)

def _broadcast_matmul_param(param, target_shape, dtype, role, numeric_dtype=None):
    np_dtype = numeric_dtype if numeric_dtype is not None else nn.DTYPE_TO_NUMPY[dtype]
    if param is None:
        return np.zeros(target_shape, dtype=np_dtype)

    values = _tensor_data_as_numeric(param) if numeric_dtype is not None else param.data
    arr = np.asarray(values, dtype=np_dtype)
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    if arr.shape == ():
        return np.broadcast_to(arr, target_shape).copy()
    if arr.ndim == 1 and len(target_shape) >= 2:
        axis_len = target_shape[-2] if role == "row" else target_shape[-1]
        if arr.shape[0] == axis_len:
            shape = (1,) * (len(target_shape) - 2)
            shape = shape + ((axis_len, 1) if role == "row" else (1, axis_len))
            return np.broadcast_to(arr.reshape(shape), target_shape).copy()
    return np.broadcast_to(arr, target_shape).copy()

def _broadcast_output_param(param, target_shape, dtype, numeric_dtype=None):
    np_dtype = numeric_dtype if numeric_dtype is not None else nn.DTYPE_TO_NUMPY[dtype]
    values = _tensor_data_as_numeric(param) if numeric_dtype is not None else param.data
    arr = np.asarray(values, dtype=np_dtype)
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    return np.broadcast_to(arr, target_shape).copy()

def _sequence_position(position, length, default=None, allow_end=False):
    if position is None:
        if default is None:
            raise ValueError("Sequence position is required")
        pos = default
    else:
        pos = int(position.data.item())
    if pos < 0:
        pos += length
    upper = length if allow_end else length - 1
    if pos < 0 or pos > upper:
        raise IndexError(f"Sequence position {pos} is out of bounds for length {length}")
    return pos

def _nms_box_to_corners(box, center_point_box):
    if center_point_box:
        x_center, y_center, width, height = box
        y1 = y_center - height / 2.0
        x1 = x_center - width / 2.0
        y2 = y_center + height / 2.0
        x2 = x_center + width / 2.0
    else:
        y1, x1, y2, x2 = box
    ymin, ymax = sorted((float(y1), float(y2)))
    xmin, xmax = sorted((float(x1), float(x2)))
    return ymin, xmin, ymax, xmax

def _nms_iou(box_a, box_b, center_point_box):
    ay1, ax1, ay2, ax2 = _nms_box_to_corners(box_a, center_point_box)
    by1, bx1, by2, bx2 = _nms_box_to_corners(box_b, center_point_box)
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter = inter_h * inter_w
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union

def _window_output_shape(size):
    if hasattr(size, "data") and size.data is not None:
        return (int(np.asarray(size.data).item()),)
    return (1,)

def _float32_to_bfloat16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)

def _cast_window_output(values, dtype):
    if dtype == "bfloat16":
        return _float32_to_bfloat16_bits(values)
    return np.asarray(values, dtype=nn.DTYPE_TO_NUMPY.get(dtype, np.float32))

def _window_values(size, periodic, dtype, kind):
    length = int(np.asarray(size.data).item())
    if length < 0:
        raise ValueError(f"Window size must be non-negative, got {length}")
    if length == 0:
        return np.empty((0,), dtype=nn.DTYPE_TO_NUMPY.get(dtype, np.float32))
    if length == 1:
        return _cast_window_output(np.ones((1,), dtype=np.float64), dtype)

    denom = length if periodic else length - 1
    n = np.arange(length, dtype=np.float64)
    if kind == "hann":
        values = np.sin(n * np.pi / denom) ** 2
    elif kind == "hamming":
        alpha = 25.0 / 46.0
        values = alpha - (1.0 - alpha) * np.cos(2.0 * np.pi * n / denom)
    elif kind == "blackman":
        values = 0.42 - 0.5 * np.cos(2.0 * np.pi * n / denom) + 0.08 * np.cos(4.0 * np.pi * n / denom)
    else:
        raise ValueError(f"Unknown window kind {kind!r}")
    return _cast_window_output(values, dtype)

def _window_values_c_first(op, size, c_func_name, kind):
    length = int(np.asarray(size.data).item())
    if length < 0:
        raise ValueError(f"Window size must be non-negative, got {length}")
    out_shape = (length,)
    if op.lib is not None and op.dtype in nn.DTYPE_MAP and size.dtype in nn.DTYPE_MAP:
        size_c = op._numpy_to_ctensor(np.ascontiguousarray(size.data), size.dtype)
        output_shape_c = (ctypes.c_int * 1)(length)
        output_c = op.lib.create_tensor(output_shape_c, 1, nn.DTYPE_MAP[op.dtype])
        getattr(op.lib, c_func_name)(size_c, output_c, ctypes.c_int(op.periodic))
        out_data = op._ctensor_to_numpy(output_c, op.dtype)
        op.lib.free_tensor(size_c)
        op.lib.free_tensor(output_c)
        return out_data
    return _window_values(size, op.periodic, op.dtype, kind)

def _numpy_dtype_to_tensor_dtype(array):
    return nn.NUMPY_TO_DTYPE.get(array.dtype.type, "float32")

def _tensor_from_numpy(array):
    array = np.asarray(array)
    dtype = _numpy_dtype_to_tensor_dtype(array)
    return Tensor(*array.shape, dtype=dtype, data=array)

def _tensor_to_numpy(value):
    if isinstance(value, Tensor):
        return value.data
    if isinstance(value, Tensor_):
        return np.zeros(value.size, dtype=nn.DTYPE_TO_NUMPY.get(value.dtype, np.float32))
    return np.asarray(value)

def _reference_feed_value(value):
    if isinstance(value, (list, tuple)):
        return [_reference_feed_value(item) for item in value]
    return _tensor_to_numpy(value)

def _graph_local_value_names(graph_proto):
    names = {value.name for value in graph_proto.input if value.name}
    names.update(value.name for value in graph_proto.initializer if value.name)
    names.update(value for node in graph_proto.node for value in node.output if value)
    return names

def _graph_external_names(graph_proto):
    local_names = _graph_local_value_names(graph_proto)
    used_names = {value for node in graph_proto.node for value in node.input if value}
    used_names.update(value.name for value in graph_proto.output if value.name)

    nested_external_names = set()
    for node in graph_proto.node:
        for attr in node.attribute:
            if attr.type == attr.GRAPH:
                nested_external_names.update(_graph_external_names(attr.g))
            elif attr.type == attr.GRAPHS:
                for nested_graph in attr.graphs:
                    nested_external_names.update(_graph_external_names(nested_graph))
    used_names.update(nested_external_names)
    return {name for name in used_names if name not in local_names}

def _graph_value_shape(value_info):
    tensor_type = value_info.type.tensor_type
    dtype = nn.onnx_dtype_mapping.get(tensor_type.elem_type, "float32")
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else 1)
    return Tensor_(*dims, dtype=dtype)

def _run_graph_proto(graph_proto, feeds, outer_scope=None):
    from onnx import helper
    from onnx.reference import ReferenceEvaluator

    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])
    evaluator = ReferenceEvaluator(model)
    input_names = {value.name for value in graph_proto.input}
    needed_names = input_names | _graph_external_names(graph_proto)
    graph_feeds = {}
    if outer_scope:
        graph_feeds.update(
            {
                name: _reference_feed_value(value)
                for name, value in outer_scope.items()
                if name in needed_names
            }
        )
    graph_feeds.update(
        {
            name: _reference_feed_value(value)
            for name, value in feeds.items()
            if name in needed_names
        }
    )
    return evaluator.run(None, graph_feeds)


__all__ = [name for name in globals() if not name.startswith("__")]
