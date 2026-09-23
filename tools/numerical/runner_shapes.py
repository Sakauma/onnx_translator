# /**
#   ******************************************************************************
#   * @file        runner_shapes.py
#   * @author      Egor Izmaylov
#   * @brief       Independently derives numerical-verifier output shapes.
#   * @details     2026.09.21  V1.0.0  Added input/schema-based shape contracts
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

"""Shape contracts for the active numerical verifier.

The resolver deliberately consumes only prepared inputs and plan attributes.  It
must not inspect the C/NPS result or call an operator's ``forward_`` method: both
would make a malformed implementation define its own expected shape.
"""

from __future__ import annotations

import math

import numpy as np
from nn import DTYPE_TO_NUMPY


class ShapeOracleError(ValueError):
    """Raised when a numerical plan has no independent, concrete shape rule."""


_SHAPE_PRESERVING = {
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "bernoulli",
    "binarizer", "bitcast", "bitwise_not", "cast", "cast_like", "ceil", "celu",
    "clip", "cos", "cosh", "cumprod", "cumsum", "dequantize_linear", "elu", "erf",
    "exp", "floor", "gelu", "group_normalization", "hard_sigmoid", "hard_swish",
    "hardmax", "identity", "instance_normalization", "isinf", "isnan",
    "leaky_relu", "log", "log_softmax", "lp_normalization", "lrn",
    "mean_variance_normalization", "mish", "neg", "not", "quantize_linear",
    "reciprocal", "relu", "reverse_sequence", "rms_normalization",
    "rotary_embedding", "round", "scatter_elements", "scatternd", "selu", "shrink",
    "sigmoid", "sign", "sin", "sinh", "softmax", "softplus", "softsign", "sqrt",
    "swish", "tan", "tanh", "tensor_scatter", "thresholded_relu", "tril", "triu",
    "trilu",
}

_BROADCASTING = {
    "add", "and", "bit_shift", "bitwise_and", "bitwise_or", "bitwise_xor", "div",
    "equal", "greater", "greater_or_equal", "less", "less_or_equal", "max", "mean",
    "min", "mod", "mul", "or", "pow", "prelu", "sub", "sum", "where", "xor",
}

_REDUCTIONS = {
    "reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_max",
    "reduce_mean", "reduce_min", "reduce_prod", "reduce_sum", "reduce_sum_square",
}

_POOLING = {"average_pool", "lp_pool", "max_pool"}


def _shape(value) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _scalar(value):
    return np.asarray(value).reshape(()).item()


def _axes(init_args, rank: int) -> tuple[int, ...]:
    axes = init_args.get("axes")
    if axes is None:
        return tuple(range(rank))
    return tuple(sorted({int(axis) % rank for axis in axes}))


def _reduce_shape(input_shape, axes, keepdims) -> tuple[int, ...]:
    axes = set(axes)
    if keepdims:
        return tuple(1 if index in axes else dim for index, dim in enumerate(input_shape))
    return tuple(dim for index, dim in enumerate(input_shape) if index not in axes)


def _matmul_shape(a_shape, b_shape) -> tuple[int, ...]:
    if len(a_shape) == 0 or len(b_shape) == 0:
        raise ShapeOracleError("MatMul inputs must have rank at least one")
    a_vector = len(a_shape) == 1
    b_vector = len(b_shape) == 1
    a = (1, a_shape[0]) if a_vector else a_shape
    b = (b_shape[0], 1) if b_vector else b_shape
    if a[-1] != b[-2]:
        raise ShapeOracleError(f"MatMul contracting dimensions differ: {a[-1]} != {b[-2]}")
    batch = np.broadcast_shapes(a[:-2], b[:-2])
    result = batch + (a[-2], b[-1])
    if a_vector:
        result = result[:-2] + result[-1:]
    if b_vector:
        result = result[:-1]
    return tuple(int(dim) for dim in result)


def _conv_output_shape(inputs, init_args, *, transpose=False) -> tuple[int, ...]:
    x_shape = _shape(inputs[0])
    w_shape = _shape(inputs[1])
    spatial_rank = len(x_shape) - 2
    strides = list(map(int, init_args.get("strides", [1] * spatial_rank)))
    dilations = list(map(int, init_args.get("dilations", [1] * spatial_rank)))
    pads = list(map(int, init_args.get("pads", [0] * (2 * spatial_rank))))
    if transpose:
        output_padding = list(map(int, init_args.get("output_padding", [0] * spatial_rank)))
        group = int(init_args.get("group", 1))
        spatial = [
            strides[i] * (x_shape[i + 2] - 1)
            + output_padding[i]
            + dilations[i] * (w_shape[i + 2] - 1)
            + 1
            - pads[i]
            - pads[i + spatial_rank]
            for i in range(spatial_rank)
        ]
        return (x_shape[0], w_shape[1] * group, *spatial)
    spatial = [
        (x_shape[i + 2] + pads[i] + pads[i + spatial_rank]
         - dilations[i] * (w_shape[i + 2] - 1) - 1) // strides[i] + 1
        for i in range(spatial_rank)
    ]
    return (x_shape[0], w_shape[0], *spatial)


def _pool_output_shape(inputs, init_args) -> tuple[int, ...]:
    x_shape = _shape(inputs[0])
    spatial_rank = len(x_shape) - 2
    kernel = list(map(int, init_args["kernel_shape"]))
    strides = list(map(int, init_args.get("strides", [1] * spatial_rank)))
    dilations = list(map(int, init_args.get("dilations", [1] * spatial_rank)))
    pads = list(map(int, init_args.get("pads", [0] * (2 * spatial_rank))))
    spatial = [
        (x_shape[i + 2] + pads[i] + pads[i + spatial_rank]
         - (dilations[i] * (kernel[i] - 1) + 1)) // strides[i] + 1
        for i in range(spatial_rank)
    ]
    return (*x_shape[:2], *spatial)


def _reshape_shape(input_shape, target) -> tuple[int, ...]:
    target = [int(value) for value in np.asarray(target).reshape(-1)]
    output = []
    unknown = None
    known_product = 1
    for index, dim in enumerate(target):
        if dim == 0:
            dim = input_shape[index]
        elif dim == -1:
            if unknown is not None:
                raise ShapeOracleError("Reshape target has more than one inferred dimension")
            unknown = index
            output.append(-1)
            continue
        if dim < 0:
            raise ShapeOracleError(f"Reshape target contains invalid dimension {dim}")
        output.append(dim)
        known_product *= dim
    if unknown is not None:
        input_product = math.prod(input_shape)
        if known_product == 0 or input_product % known_product:
            raise ShapeOracleError("Reshape inferred dimension is not integral")
        output[unknown] = input_product // known_product
    return tuple(output)


def _slice_length(dim, start, end, step) -> int:
    return len(range(*slice(int(start), int(end), int(step)).indices(int(dim))))


def _nms_selected_count(inputs, init_args) -> int:
    boxes = np.asarray(inputs[0], dtype=np.float64)
    scores = np.asarray(inputs[1], dtype=np.float64)
    max_output = max(0, int(_scalar(inputs[2])))
    iou_threshold = float(_scalar(inputs[3]))
    score_threshold = float(_scalar(inputs[4]))
    center = bool(init_args.get("center_point_box", 0))

    def corners(box):
        if not center:
            y1, x1, y2, x2 = box
            return min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)
        x, y, w, h = box
        return y - h / 2.0, x - w / 2.0, y + h / 2.0, x + w / 2.0

    def iou(lhs, rhs):
        ly1, lx1, ly2, lx2 = corners(lhs)
        ry1, rx1, ry2, rx2 = corners(rhs)
        inter = max(0.0, min(ly2, ry2) - max(ly1, ry1)) * max(
            0.0, min(lx2, rx2) - max(lx1, rx1)
        )
        lhs_area = max(0.0, ly2 - ly1) * max(0.0, lx2 - lx1)
        rhs_area = max(0.0, ry2 - ry1) * max(0.0, rx2 - rx1)
        union = lhs_area + rhs_area - inter
        return 0.0 if union <= 0.0 else inter / union

    count = 0
    for batch in range(scores.shape[0]):
        for class_index in range(scores.shape[1]):
            candidates = [
                index for index, score in enumerate(scores[batch, class_index])
                if score >= score_threshold
            ]
            candidates.sort(key=lambda index: (-scores[batch, class_index, index], index))
            selected = []
            while candidates and len(selected) < max_output:
                current = candidates.pop(0)
                selected.append(current)
                candidates = [
                    index for index in candidates
                    if iou(boxes[batch, current], boxes[batch, index]) <= iou_threshold
                ]
            count += len(selected)
    return count


def resolve_output_shapes(op_name: str, inputs, init_args=None) -> tuple[tuple[int, ...], ...]:
    """Return every logical output shape for one prepared numerical sample."""

    init_args = init_args or {}
    present = [value for value in inputs if value is not None]
    if op_name in _SHAPE_PRESERVING:
        return (_shape(inputs[0]),)
    if op_name in _BROADCASTING:
        return (tuple(np.broadcast_shapes(*(_shape(value) for value in present))),)
    if op_name in _REDUCTIONS:
        input_shape = _shape(inputs[0])
        return (_reduce_shape(input_shape, _axes(init_args, len(input_shape)), int(init_args.get("keepdims", 1))),)
    if op_name in _POOLING:
        return (_pool_output_shape(inputs, init_args),)

    if op_name in {"conv2d", "conv_integer", "deform_conv"}:
        return (_conv_output_shape(inputs, init_args),)
    if op_name == "qlinear_conv":
        conv_inputs = [inputs[0], inputs[3]]
        return (_conv_output_shape(conv_inputs, init_args),)
    if op_name == "conv_transpose":
        return (_conv_output_shape(inputs, init_args, transpose=True),)
    if op_name in {"matmul", "matmul_integer"}:
        return (_matmul_shape(_shape(inputs[0]), _shape(inputs[1])),)
    if op_name == "qlinear_matmul":
        return (_matmul_shape(_shape(inputs[0]), _shape(inputs[3])),)
    if op_name == "gemm":
        a = _shape(inputs[0])
        b = _shape(inputs[1])
        m = a[1] if int(init_args.get("transA", 0)) else a[0]
        n = b[0] if int(init_args.get("transB", 0)) else b[1]
        return ((m, n),)
    if op_name == "einsum":
        equation = init_args.get("equation")
        if equation == "ij,jk->ik":
            return ((_shape(inputs[0])[0], _shape(inputs[1])[1]),)
        raise ShapeOracleError(f"unsupported Einsum equation: {equation!r}")
    if op_name == "attention":
        query = _shape(inputs[0])
        value = _shape(inputs[2])
        return ((*query[:-1], value[-1]),)

    if op_name == "argmax" or op_name == "argmin":
        shape = _shape(inputs[0])
        axis = int(init_args.get("axis", 0)) % len(shape)
        return (_reduce_shape(shape, (axis,), int(init_args.get("keepdims", 1))),)
    if op_name == "det":
        return (_shape(inputs[0])[:-2],)
    if op_name == "size":
        return ((),)

    if op_name == "flatten":
        shape = _shape(inputs[0])
        axis = int(init_args.get("axis", 1))
        axis = axis + len(shape) if axis < 0 else axis
        return ((math.prod(shape[:axis]), math.prod(shape[axis:])),)
    if op_name == "reshape":
        return (_reshape_shape(_shape(inputs[0]), inputs[1]),)
    if op_name == "squeeze":
        shape = _shape(inputs[0])
        axes = init_args.get("axes")
        if axes is None and len(inputs) > 1 and inputs[1] is not None:
            axes = np.asarray(inputs[1]).reshape(-1).tolist()
        remove = {int(axis) % len(shape) for axis in axes} if axes is not None else {
            index for index, dim in enumerate(shape) if dim == 1
        }
        return (tuple(dim for index, dim in enumerate(shape) if index not in remove),)
    if op_name == "unsqueeze":
        shape = list(_shape(inputs[0]))
        axes = init_args.get("axes")
        if axes is None and len(inputs) > 1 and inputs[1] is not None:
            axes = np.asarray(inputs[1]).reshape(-1).tolist()
        output_rank = len(shape) + len(axes)
        normalized = sorted(int(axis) % output_rank for axis in axes)
        for axis in normalized:
            shape.insert(axis, 1)
        return (tuple(shape),)
    if op_name == "transpose":
        shape = _shape(inputs[0])
        perm = init_args.get("perm", list(reversed(range(len(shape)))))
        return (tuple(shape[int(axis)] for axis in perm),)
    if op_name == "tile":
        shape = _shape(inputs[0])
        repeats = np.asarray(inputs[1]).reshape(-1)
        return (tuple(dim * int(repeat) for dim, repeat in zip(shape, repeats)),)
    if op_name == "concat":
        shapes = [_shape(value) for value in present]
        axis = int(init_args.get("axis", 0)) % len(shapes[0])
        output = list(shapes[0])
        output[axis] = sum(shape[axis] for shape in shapes)
        return (tuple(output),)
    if op_name == "expand":
        return (tuple(int(value) for value in np.asarray(inputs[1]).reshape(-1)),)
    if op_name == "pad":
        shape = _shape(inputs[0])
        pads = [int(value) for value in np.asarray(inputs[1]).reshape(-1)]
        rank = len(shape)
        return (tuple(shape[i] + pads[i] + pads[i + rank] for i in range(rank)),)
    if op_name == "center_crop_pad":
        shape = list(_shape(inputs[0]))
        target = [int(value) for value in np.asarray(inputs[1]).reshape(-1)]
        axes = init_args.get("axes")
        if axes is None:
            axes = range(len(shape))
        for axis, dim in zip(axes, target):
            shape[int(axis) % len(shape)] = dim
        return (tuple(shape),)
    if op_name == "slice":
        shape = list(_shape(inputs[0]))
        starts = np.asarray(inputs[1]).reshape(-1)
        ends = np.asarray(inputs[2]).reshape(-1)
        axes = np.arange(len(starts)) if len(inputs) < 4 or inputs[3] is None else np.asarray(inputs[3]).reshape(-1)
        steps = np.ones(len(starts), dtype=np.int64) if len(inputs) < 5 or inputs[4] is None else np.asarray(inputs[4]).reshape(-1)
        for start, end, axis, step in zip(starts, ends, axes, steps):
            axis = int(axis) % len(shape)
            shape[axis] = _slice_length(shape[axis], start, end, step)
        return (tuple(shape),)
    if op_name == "depth_to_space":
        n, c, h, w = _shape(inputs[0])
        block = int(init_args["blocksize"])
        return ((n, c // (block * block), h * block, w * block),)
    if op_name == "space_to_depth":
        n, c, h, w = _shape(inputs[0])
        block = int(init_args["blocksize"])
        return ((n, c * block * block, h // block, w // block),)

    if op_name == "gather":
        data = _shape(inputs[0])
        indices = _shape(inputs[1])
        axis = int(init_args.get("axis", 0)) % len(data)
        return ((data[:axis] + indices + data[axis + 1:]),)
    if op_name == "gather_elements":
        return (_shape(inputs[1]),)
    if op_name == "gathernd":
        data = _shape(inputs[0])
        indices = _shape(inputs[1])
        batch_dims = int(init_args.get("batch_dims", 0))
        index_depth = indices[-1]
        return ((indices[:-1] + data[batch_dims + index_depth:]),)
    if op_name == "one_hot":
        indices = list(_shape(inputs[0]))
        depth = int(_scalar(inputs[1]))
        output_rank = len(indices) + 1
        axis = int(init_args.get("axis", -1)) % output_rank
        indices.insert(axis, depth)
        return (tuple(indices),)

    if op_name == "global_average_pool" or op_name == "global_max_pool" or op_name == "global_lp_pool":
        shape = _shape(inputs[0])
        return ((shape[:2] + (1,) * (len(shape) - 2)),)
    if op_name == "max_roi_pool":
        x = _shape(inputs[0])
        rois = _shape(inputs[1])
        return ((rois[0], x[1], *tuple(map(int, init_args["pooled_shape"]))),)
    if op_name == "roi_align":
        x = _shape(inputs[0])
        rois = _shape(inputs[1])
        return ((rois[0], x[1], int(init_args["output_height"]), int(init_args["output_width"])),)
    if op_name == "grid_sample":
        x = _shape(inputs[0])
        grid = _shape(inputs[1])
        return ((x[0], x[1], *grid[1:-1]),)
    if op_name == "affine_grid":
        size = [int(value) for value in np.asarray(inputs[1]).reshape(-1)]
        return ((size[0], *size[2:], len(size) - 2),)
    if op_name == "max_unpool":
        shape = _shape(inputs[0])
        if len(inputs) > 2 and inputs[2] is not None:
            return (tuple(int(value) for value in np.asarray(inputs[2]).reshape(-1)),)
        rank = len(shape) - 2
        kernel = list(map(int, init_args["kernel_shape"]))
        strides = list(map(int, init_args.get("strides", kernel)))
        pads = list(map(int, init_args.get("pads", [0] * (2 * rank))))
        spatial = [
            (shape[i + 2] - 1) * strides[i] + kernel[i] - pads[i] - pads[i + rank]
            for i in range(rank)
        ]
        return ((*shape[:2], *spatial),)
    if op_name == "col2im":
        x = _shape(inputs[0])
        image = [int(value) for value in np.asarray(inputs[1]).reshape(-1)]
        block = [int(value) for value in np.asarray(inputs[2]).reshape(-1)]
        return ((x[0], x[1] // math.prod(block), *image),)

    if op_name == "resize":
        if len(inputs) > 3 and np.asarray(inputs[3]).size:
            return (tuple(int(value) for value in np.asarray(inputs[3]).reshape(-1)),)
        scales = np.asarray(inputs[2]).reshape(-1)
        return (tuple(int(math.floor(dim * float(scale))) for dim, scale in zip(_shape(inputs[0]), scales)),)
    if op_name == "constant_of_shape":
        return (tuple(int(value) for value in np.asarray(inputs[0]).reshape(-1)),)
    if op_name == "eye_like":
        return (_shape(inputs[0]),)
    if op_name in {"random_uniform", "random_normal"}:
        return (tuple(map(int, init_args["shape"])),)
    if op_name in {"random_uniform_like", "random_normal_like"}:
        return (_shape(inputs[0]),)
    if op_name == "multinomial":
        return ((_shape(inputs[0])[0], int(init_args.get("sample_size", 1))),)
    if op_name in {"hann_window", "hamming_window", "blackman_window"}:
        return ((int(_scalar(inputs[0])),),)
    if op_name == "mel_weight_matrix":
        bins = int(_scalar(inputs[0]))
        dft_length = int(_scalar(inputs[1]))
        return ((dft_length // 2 + 1, bins),)

    if op_name == "dft":
        shape = list(_shape(inputs[0]))
        axis = int(init_args.get("axis", 1)) % len(shape)
        length = int(_scalar(inputs[1])) if len(inputs) > 1 and inputs[1] is not None else shape[axis]
        inverse = int(init_args.get("inverse", 0))
        onesided = int(init_args.get("onesided", 0))
        shape[axis] = length // 2 + 1 if onesided and not inverse else length
        shape[-1] = 1 if onesided and inverse else 2
        return (tuple(shape),)
    if op_name == "stft":
        signal = _shape(inputs[0])
        frame_step = int(_scalar(inputs[1]))
        frame_length = int(_scalar(inputs[3])) if len(inputs) > 3 and inputs[3] is not None else _shape(inputs[2])[0]
        frames = (signal[-2] - frame_length) // frame_step + 1
        frequencies = frame_length // 2 + 1 if int(init_args.get("onesided", 1)) else frame_length
        return ((*signal[:-2], frames, frequencies, 2),)

    if op_name == "range":
        start, limit, delta = (_scalar(inputs[index]) for index in range(3))
        return ((int(np.arange(start, limit, delta).size),),)
    if op_name == "nonzero":
        value = np.asarray(inputs[0])
        return ((value.ndim, int(np.count_nonzero(value))),)
    if op_name == "compress":
        value = np.asarray(inputs[0])
        condition = np.asarray(inputs[1], dtype=np.bool_).reshape(-1)
        axis = init_args.get("axis")
        available = value.size if axis is None else value.shape[int(axis) % value.ndim]
        count = int(np.count_nonzero(condition[:available]))
        if axis is None:
            return ((count,),)
        shape = list(value.shape)
        shape[int(axis) % value.ndim] = count
        return (tuple(shape),)
    if op_name == "non_max_suppression":
        return ((_nms_selected_count(inputs, init_args), 3),)

    if op_name == "negative_log_likelihood_loss":
        target_shape = _shape(inputs[1])
        return ((target_shape if init_args.get("reduction", "mean") == "none" else ()),)
    if op_name == "softmax_cross_entropy_loss":
        target_shape = _shape(inputs[1])
        loss = target_shape if init_args.get("reduction", "mean") == "none" else ()
        outputs = [loss]
        if int(init_args.get("emit_log_prob", 0)):
            outputs.append(_shape(inputs[0]))
        return tuple(outputs)

    if op_name == "dropout":
        shape = _shape(inputs[0])
        return (shape, shape)
    if op_name == "batch_normalization":
        output = [_shape(inputs[0])]
        if int(init_args.get("training_mode", 0)):
            output.extend((_shape(inputs[1]), _shape(inputs[2])))
        return tuple(output)
    if op_name == "layer_normalization":
        input_shape = _shape(inputs[0])
        output = [input_shape]
        if int(init_args.get("emit_stats", 0)):
            axis = int(init_args.get("axis", -1)) % len(input_shape)
            stats_shape = input_shape[:axis] + (1,) * (len(input_shape) - axis)
            output.extend((stats_shape, stats_shape))
        return tuple(output)
    if op_name == "dynamic_quantize_linear":
        return (_shape(inputs[0]), (), ())
    if op_name == "topk":
        shape = list(_shape(inputs[0]))
        axis = int(init_args.get("axis", -1)) % len(shape)
        shape[axis] = int(_scalar(inputs[1]))
        return (tuple(shape), tuple(shape))
    if op_name == "split":
        shape = list(_shape(inputs[0]))
        axis = int(init_args.get("axis", 0)) % len(shape)
        if len(inputs) > 1 and inputs[1] is not None:
            split = [int(value) for value in np.asarray(inputs[1]).reshape(-1)]
        else:
            count = int(init_args.get("num_outputs", 1))
            if shape[axis] % count:
                raise ShapeOracleError("Split dimension is not divisible by num_outputs")
            split = [shape[axis] // count] * count
        outputs = []
        for length in split:
            piece = list(shape)
            piece[axis] = length
            outputs.append(tuple(piece))
        return tuple(outputs)
    if op_name == "unique":
        value = np.asarray(inputs[0])
        axis = init_args.get("axis")
        if axis is None:
            unique_len = int(np.unique(value.reshape(-1)).size)
            inverse_shape = (value.size,)
            values_shape = (unique_len,)
        else:
            axis = int(axis) % value.ndim
            unique_len = int(np.unique(value, axis=axis).shape[axis])
            inverse_shape = (value.shape[axis],)
            values_shape = list(value.shape)
            values_shape[axis] = unique_len
            values_shape = tuple(values_shape)
        return (values_shape, (unique_len,), inverse_shape, (unique_len,))
    if op_name in {"rnn", "gru", "lstm"}:
        x = _shape(inputs[0])
        hidden = int(init_args["hidden_size"])
        directions = 2 if init_args.get("direction", "forward") == "bidirectional" else 1
        if int(init_args.get("layout", 0)):
            batch, sequence = x[0], x[1]
            outputs = [(batch, sequence, directions, hidden), (batch, directions, hidden)]
        else:
            sequence, batch = x[0], x[1]
            outputs = [(sequence, directions, batch, hidden), (directions, batch, hidden)]
        if op_name == "lstm":
            outputs.append(outputs[1])
        return tuple(outputs)

    raise ShapeOracleError(f"no independent output-shape rule for numerical op {op_name!r}")


def resolve_output_dtypes(op_name, out_dtype, init_args, output_count):
    """Derive public output storage dtypes from the plan and operator contract.

    CUDA may use a different transport dtype; this contract describes the NPS
    result before any conversion to that transport format.
    """

    init_args = init_args or {}
    if op_name == "topk":
        dtypes = (out_dtype, "int64")
    elif op_name == "dropout":
        dtypes = (out_dtype, "bool")
    elif op_name == "dynamic_quantize_linear":
        dtypes = ("uint8", "float32", "uint8")
    elif op_name == "unique":
        dtypes = (out_dtype, "int64", "int64", "int64")
    elif op_name == "layer_normalization" and int(init_args.get("emit_stats", 0)):
        stash_type = int(init_args.get("stash_type", 1))
        stash_dtype = {1: "float32", 16: "bfloat16"}.get(stash_type)
        if stash_dtype is None:
            raise ShapeOracleError(f"unsupported LayerNormalization stash_type {stash_type}")
        dtypes = (out_dtype, stash_dtype, stash_dtype)
    else:
        # BatchNormalization training, recurrent states, Split pieces, and
        # SoftmaxCrossEntropyLoss log_prob share the declared primary dtype.
        dtypes = (out_dtype,) * output_count
    if len(dtypes) != output_count:
        raise ShapeOracleError(
            f"{op_name} output dtype count mismatch: expected {output_count}, got {len(dtypes)}"
        )
    for dtype in dtypes:
        if dtype not in DTYPE_TO_NUMPY:
            raise ShapeOracleError(f"{op_name} has unknown output dtype {dtype!r}")
    return dtypes


def validate_nps_output_shapes(op_name, nps_result, expected_shapes, expected_dtypes=None):
    """Validate NPS output count, rank, dimensions, and storage dtypes."""

    if op_name == "topk":
        actual = [nps_result.output, nps_result.topk_indices]
    elif isinstance(nps_result.output, (list, tuple)):
        actual = list(nps_result.output)
    else:
        actual = [nps_result.output]
    if len(actual) != len(expected_shapes):
        raise ShapeOracleError(
            f"{op_name} output count mismatch: expected {len(expected_shapes)}, got {len(actual)}"
        )
    if expected_dtypes is not None and len(expected_dtypes) != len(expected_shapes):
        raise ShapeOracleError(f"{op_name} output dtype contract length mismatch")
    for index, (value, expected) in enumerate(zip(actual, expected_shapes)):
        actual_shape = _shape(value)
        if actual_shape != tuple(expected):
            raise ShapeOracleError(
                f"{op_name} output {index} shape mismatch: expected {tuple(expected)}, got {actual_shape}"
            )
        if expected_dtypes is not None:
            expected_dtype = np.dtype(DTYPE_TO_NUMPY[expected_dtypes[index]])
            actual_dtype = np.asarray(value).dtype
            if actual_dtype != expected_dtype:
                raise ShapeOracleError(
                    f"{op_name} NPS output dtype mismatch at output {index}: "
                    f"expected {expected_dtype}, got {actual_dtype}"
                )
