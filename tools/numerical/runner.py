# /**
#   ******************************************************************************
#   * @file        runner.py
#   * @author      Egor Izmaylov
#   * @brief       执行单个算子的数值验证计划，包括输入准备、参数打包和结果比较。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import os
import traceback

import numpy as np

import nn
from nn import Tensor

from .compare import check_accuracy
from .cuda import run_cuda_ground_truth
from .data import generate_random_data, random_uniform_like_reference
from .dtype import from_float32, quantize_to_dtype_float32, to_float32


def _slice_io_values(init_args, input_shape):
    rank = len(input_shape)
    starts = np.asarray(init_args.get("starts_value", [0] * rank), dtype=np.int64).reshape(-1)
    ends = np.asarray(init_args.get("ends_value", list(input_shape)), dtype=np.int64).reshape(-1)
    axes = np.asarray(init_args.get("axes_value", list(range(len(starts)))), dtype=np.int64).reshape(-1)
    steps = np.asarray(init_args.get("steps_value", [1] * len(starts)), dtype=np.int64).reshape(-1)
    return starts, ends, axes, steps


def _normalize_slice_parameters(input_shape, starts, ends, axes, steps):
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


def verify_op(op_cls, op_name, shapes, dtypes, out_dtype, init_args=None, iterations=5):
    init_args = init_args or {}
    print(f"🧪 Testing {op_name.upper()}: {dtypes} -> {out_dtype}")
    
    atol, rtol = 1e-4, 1e-4
    if "float16" in out_dtype: atol, rtol = 0.01, 0.01 
    if "bfloat16" in out_dtype: atol, rtol = 0.1, 0.02
    if "float8" in out_dtype: atol, rtol = 0.1, 0.1    
    if "int" in out_dtype: atol, rtol = 0, 0
    if op_name == "cos": atol = max(atol, 0.02)
    if op_name == "einsum":
        atol, rtol = max(atol, 1e-2), max(rtol, 1e-3)

    pass_cnt = 0
    stats_abs = []
    stats_rel = []
    
    for i in range(iterations):
        # 1. 生成数据
        inputs_np = []
        for s, d in zip(shapes, dtypes):
            if s is None: inputs_np.append(None)
            else: inputs_np.append(generate_random_data(s, d))

        if op_name == "clip":
            inputs_np[1] = from_float32(np.full(shapes[1], -1.0, dtype=np.float32), dtypes[1])
            inputs_np[2] = from_float32(np.full(shapes[2], 1.0, dtype=np.float32), dtypes[2])

        if op_name == "sqrt":
            # 主路径：保证非负，避免 NaN 干扰对齐
            inputs_np[0] = from_float32(np.abs(to_float32(inputs_np[0], dtypes[0])), dtypes[0])

        if op_name in {"equal", "greater", "less", "greater_or_equal", "less_or_equal"}:
            # 比较类混合精度计划使用有限且包含等值/大小关系的样本，避免 float8 随机 NaN 掩盖比较语义。
            total = int(np.prod(shapes[0]))
            base = np.linspace(-3.0, 3.0, total, dtype=np.float32).reshape(shapes[0])
            offsets = (((np.arange(total, dtype=np.int32) % 5) - 2).astype(np.float32) * 0.25).reshape(shapes[1])
            inputs_np[0] = from_float32(base, dtypes[0])
            inputs_np[1] = from_float32(base + offsets, dtypes[1])

        if op_name == "isinf":
            # IsInf 使用显式 +/-Inf 与有限值样本，覆盖 detect_positive/detect_negative 标志。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-4.0, 4.0, total, dtype=np.float32).reshape(shapes[0])
            flat = values.reshape(-1)
            flat[0] = np.inf
            flat[1] = -np.inf
            flat[2] = 0.0
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "identity":
            # Identity 使用有限可量化样本，覆盖低精度位模式经 C 后端原样传递的路径。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-3.0, 3.0, total, dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "where":
            # Where 使用固定 bool 条件和两组不同值，覆盖条件选择、广播前物化和低精度写回。
            total = int(np.prod(shapes[0]))
            cond = ((np.arange(total).reshape(shapes[0]) % 3) != 1)
            x_values = np.linspace(-2.5, 2.5, total, dtype=np.float32).reshape(shapes[1])
            y_values = np.linspace(3.0, -3.0, total, dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = cond.astype(np.bool_)
            inputs_np[1] = from_float32(x_values, dtypes[1])
            inputs_np[2] = from_float32(y_values, dtypes[2])

        if op_name in {"bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "bit_shift"}:
            # 位运算使用显式 int32 样本，覆盖正数、负数和高位位模式，避免随机大位移触发未定义语义。
            base_values = np.array(
                [
                    0, 1, -1, 2,
                    -2, 7, -8, 15,
                    16, -31, 63, -64,
                    127, -128, 255, -256,
                ],
                dtype=np.int32,
            ).reshape(shapes[0])
            inputs_np[0] = base_values
            if len(inputs_np) > 1:
                if op_name == "bit_shift":
                    shifts = (np.arange(int(np.prod(shapes[1])), dtype=np.int32) % 5).reshape(shapes[1])
                    inputs_np[1] = shifts
                else:
                    rhs = np.array(
                        [
                            3, 5, -7, 9,
                            -11, 13, 17, -19,
                            23, -29, 31, -37,
                            41, -43, 47, -53,
                        ],
                        dtype=np.int32,
                    ).reshape(shapes[1])
                    inputs_np[1] = rhs

        if op_name == "gather":
            M, N = shapes[0]      # data shape (M,N)
            idx_shape = shapes[1] # indices shape (I,)
            inputs_np[1] = np.random.randint(0, M, size=idx_shape).astype(np.int64)
          
        if op_name in ["quantize_linear", "dequantize_linear"]:
            if inputs_np[1] is not None: inputs_np[1] = np.abs(inputs_np[1]) + 1e-4
            if inputs_np[2] is not None:
                inputs_np[2] = np.round(inputs_np[2])
                if op_name == "quantize_linear": inputs_np[2] = np.clip(inputs_np[2], -128, 127)
        if op_name == "scatternd":
            M, N = shapes[0]       # data: (M,N)
            I, K = shapes[1]       # indices: (I,2)
            assert K == 2
            rng = np.random.default_rng(0)
            flat = rng.choice(M * N, size=I, replace=False)
            rows = flat // N
            cols = flat % N
            inputs_np[1] = np.stack([rows, cols], axis=1).astype(np.int64)

        if op_name == "tensor_scatter":
            # TensorScatter 使用固定有限样本，重点验证 cache 坐标映射和低精度量化后的位置语义。
            cache_values = np.linspace(-2.0, 2.0, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            update_values = np.linspace(3.0, -3.0, int(np.prod(shapes[1])), dtype=np.float32).reshape(shapes[1])
            inputs_np[0] = from_float32(cache_values, dtypes[0])
            inputs_np[1] = from_float32(update_values, dtypes[1])
            inputs_np[2] = np.asarray(init_args.get("write_indices_value", [0] * shapes[0][0]), dtype=np.int64)

        if op_name == "gather_elements":
            # 主路径：data=(M,N), indices=(M,N), axis=1
            M, N = shapes[0]
            inputs_np[1] = np.random.randint(0, N, size=(M, N)).astype(np.int64)

        if op_name == "gathernd":
            # 简化主路径：data=(M,N), indices=(I,2) -> out=(I,)
            M, N = shapes[0]
            I, K = shapes[1]
            assert K == 2
            rows = np.random.randint(0, M, size=I, dtype=np.int64)
            cols = np.random.randint(0, N, size=I, dtype=np.int64)
            inputs_np[1] = np.stack([rows, cols], axis=1).astype(np.int64)
        if op_name == "reduce_prod":
            inputs_np[0] = from_float32(np.clip(to_float32(inputs_np[0], dtypes[0]), -1.1, 1.1), dtypes[0])

        if op_name == "cumprod":
            # 累计乘积使用温和的正数样本，避免低精度随机值导致指数级误差放大。
            total = int(np.prod(shapes[0]))
            values = np.linspace(0.75, 1.25, total, dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name in {"reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square"}:
            # 公式归约使用有限样本，避免 LogSum 的非正输入和低精度随机 NaN 干扰主语义验证。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-1.0, 1.0, total, dtype=np.float32).reshape(shapes[0])
            if op_name == "reduce_log_sum":
                values = np.abs(values) + 0.25
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "nonzero":
            # 保证既有 0 也有非 0，避免输出全空或全满太极端
            x = to_float32(inputs_np[0], dtypes[0]).astype(np.float32)
            mask = np.random.rand(*x.shape) < 0.35
            x[mask] = 0.0
            x[~mask] = np.where(np.abs(x[~mask]) < 1e-3, 1.0, x[~mask])
            inputs_np[0] = from_float32(x, dtypes[0])

        if op_name == "argmin" or op_name == "argmax":
            # Arg 类计划固定为 2D + axis=1，并使用可量化后仍无 tie 的行内递增数据。
            M, N = shapes[0]
            row_offsets = np.arange(M, dtype=np.float32).reshape(M, 1) * 0.5
            col_values = np.arange(N, dtype=np.float32).reshape(1, N) - (N // 2)
            inputs_np[0] = from_float32(row_offsets + col_values, dtypes[0])

        if op_name == "resize":
            # inputs: x, roi, scales, sizes
            target_sizes = init_args.get("sizes_value", list(shapes[0]))
            inputs_np[1] = np.array([], dtype=np.float32)   # roi
            inputs_np[2] = np.array([], dtype=np.float32)   # scales
            inputs_np[3] = np.array(target_sizes, dtype=np.int64)

        if op_name == "affine_grid":
            # AffineGrid 使用稳定的仿射矩阵样本，避免随机低精度位模式生成 NaN 或极端网格。
            size_value = list(map(int, init_args.get("size_value", [2, 1, 3, 4])))
            inputs_np[1] = np.array(size_value, dtype=np.int64)
            if len(size_value) == 4:
                theta_values = np.array(
                    [
                        [[1.0, 0.0, 0.1], [0.0, 1.0, -0.2]],
                        [[0.8, 0.1, 0.0], [-0.1, 0.9, 0.2]],
                    ],
                    dtype=np.float32,
                )
            elif len(size_value) == 5:
                theta_values = np.array(
                    [
                        [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.3]],
                        [[0.8, 0.1, 0.0, 0.0], [-0.1, 0.9, 0.1, 0.2], [0.0, 0.2, 0.7, -0.1]],
                    ],
                    dtype=np.float32,
                )
            else:
                raise ValueError(f"AffineGrid size_value must have rank 4 or 5, got {size_value}")
            inputs_np[0] = from_float32(theta_values[: size_value[0]], dtypes[0])

        if op_name == "grid_sample":
            # GridSample 使用固定有限样本，覆盖归一化坐标、reflection padding 和低精度写回。
            x_values = np.linspace(-1.5, 1.5, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            grid_values = np.array(
                [
                    [
                        [[-1.15, -0.85], [-0.35, -0.40], [0.25, -0.10], [1.15, 0.05]],
                        [[-0.95, 0.45], [-0.25, 0.15], [0.50, 0.35], [1.05, 0.70]],
                        [[-1.20, 1.10], [-0.45, 0.95], [0.35, 0.80], [1.25, 1.15]],
                    ]
                ],
                dtype=np.float32,
            )
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(grid_values, dtypes[1])

        if op_name == "lrn":
            # LRN 使用固定有限样本，覆盖跨通道平方和窗口和低精度写回主路径。
            values = np.linspace(-1.2, 1.3, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name in {"expand", "flatten", "reshape", "squeeze", "unsqueeze", "transpose", "pad", "center_crop_pad", "depth_to_space", "space_to_depth"}:
            # 形状变换类算子使用有限且可量化的固定样本，避免随机 float8 NaN 干扰位模式验证。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-3.0, 3.0, total, dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name in {"cast", "cast_like"}:
            # Cast 类计划使用包含负数、零和小数的有限样本，覆盖向零截断、bool 和低精度写回主路径。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-7.5, 7.5, total, dtype=np.float32).reshape(shapes[0])
            values.reshape(-1)[total // 2] = 0.0
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "bitcast":
            # BitCast 必须保留原始位模式，输入按 dtype 容器直接构造，避免数值转换改变字节。
            input_dtype = nn.DTYPE_TO_NUMPY[dtypes[0]]
            elem_size = np.dtype(input_dtype).itemsize
            raw = ((np.arange(int(np.prod(shapes[0])) * elem_size, dtype=np.uint16) * 37 + 11) & 0xFF).astype(np.uint8)
            inputs_np[0] = raw.view(input_dtype).reshape(shapes[0]).copy()

        if op_name == "rms_normalization":
            # RMSNormalization 使用稳定有限样本，覆盖 scale 单向广播和低精度 stash_type=FLOAT 主路径。
            total = int(np.prod(shapes[0]))
            x_values = np.linspace(-2.0, 2.0, total, dtype=np.float32).reshape(shapes[0])
            scale_total = int(np.prod(shapes[1]))
            scale_values = np.linspace(0.5, 1.5, scale_total, dtype=np.float32).reshape(shapes[1])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(scale_values, dtypes[1])

        if op_name == "mean_variance_normalization":
            # MeanVarianceNormalization 使用按通道不同分布的固定样本，覆盖 axes 归约和低精度写回。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-2.4, 2.8, total, dtype=np.float32).reshape(shapes[0])
            if len(shapes[0]) >= 2:
                channel_offsets = np.linspace(-0.35, 0.45, shapes[0][1], dtype=np.float32).reshape(
                    (1, shapes[0][1]) + (1,) * (len(shapes[0]) - 2)
                )
                values = values + channel_offsets
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "batch_normalization":
            # BatchNormalization 使用固定有限样本，覆盖推理公式、通道参数广播和低精度写回主路径。
            x_values = np.linspace(-1.2, 1.3, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            scale_values = np.array([1.0, 0.5, 1.5], dtype=np.float32).reshape(shapes[1])
            bias_values = np.array([0.1, -0.2, 0.3], dtype=np.float32).reshape(shapes[2])
            mean_values = np.array([0.0, 0.2, -0.1], dtype=np.float32).reshape(shapes[3])
            var_values = np.array([1.0, 0.5, 2.0], dtype=np.float32).reshape(shapes[4])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(scale_values, dtypes[1])
            inputs_np[2] = from_float32(bias_values, dtypes[2])
            inputs_np[3] = from_float32(mean_values, dtypes[3])
            inputs_np[4] = from_float32(var_values, dtypes[4])

        if op_name == "instance_normalization":
            # InstanceNormalization 使用每通道不同 scale/bias 的固定样本，覆盖 per-instance spatial 归一化。
            x_values = np.linspace(-1.4, 1.6, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            scale_values = np.array([1.0, 0.5, 1.5], dtype=np.float32).reshape(shapes[1])
            bias_values = np.array([0.1, -0.2, 0.3], dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(scale_values, dtypes[1])
            inputs_np[2] = from_float32(bias_values, dtypes[2])

        if op_name == "layer_normalization":
            # LayerNormalization 使用最后一维 scale/bias，覆盖当前 C 后端承载的 axis=-1 主路径和低精度写回。
            x_values = np.linspace(-1.8, 1.4, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            scale_values = np.array([1.0, 0.5, 1.5, -0.5], dtype=np.float32).reshape(shapes[1])
            bias_values = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(scale_values, dtypes[1])
            inputs_np[2] = from_float32(bias_values, dtypes[2])

        if op_name == "lp_normalization":
            # LpNormalization 使用固定有限样本，覆盖 p=1/p=2 下按 axis 归一化且保留输入符号的路径。
            values = np.linspace(-1.5, 1.7, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            values = np.where(np.abs(values) < 0.05, values + 0.25, values)
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "group_normalization":
            # GroupNormalization 使用每组不同分布的固定样本，覆盖 group 均值方差、通道仿射和低精度写回。
            x_values = np.linspace(-1.6, 1.8, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            scale_values = np.array([1.0, 0.5, 1.5, -0.75], dtype=np.float32).reshape(shapes[1])
            bias_values = np.array([0.1, -0.2, 0.3, -0.1], dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(scale_values, dtypes[1])
            inputs_np[2] = from_float32(bias_values, dtypes[2])

        if op_name == "rotary_embedding":
            # RotaryEmbedding 使用固定角度 cache 和 position_ids，覆盖官方 full-rotation 主路径和低精度写回。
            total = int(np.prod(shapes[0]))
            x_values = np.linspace(-1.5, 1.5, total, dtype=np.float32).reshape(shapes[0])
            half = shapes[1][-1]
            max_pos = shapes[1][0]
            angles = np.linspace(0.0, 1.0, max_pos * half, dtype=np.float32).reshape(shapes[1])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(np.cos(angles).astype(np.float32), dtypes[1])
            inputs_np[2] = from_float32(np.sin(angles).astype(np.float32), dtypes[2])
            inputs_np[3] = np.asarray(init_args.get("position_ids_value", [[0, 1, 2], [3, 4, 5]]), dtype=np.int64)

        if op_name == "col2im":
            # Col2Im 使用固定列块样本，明确覆盖重叠累加回图像张量的主路径。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-2.0, 2.0, total, dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = np.asarray(init_args.get("image_shape_value", [3, 3]), dtype=np.int64)
            inputs_np[2] = np.asarray(init_args.get("block_shape_value", [2, 2]), dtype=np.int64)

        if op_name == "deform_conv":
            # DeformConv 使用小幅 offset 和有限 mask，覆盖双线性采样、bias 和低精度写回路径。
            x_values = np.linspace(-1.2, 1.2, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            w_values = np.linspace(-0.7, 0.8, int(np.prod(shapes[1])), dtype=np.float32).reshape(shapes[1])
            offset_values = np.linspace(-0.25, 0.25, int(np.prod(shapes[2])), dtype=np.float32).reshape(shapes[2])
            bias_values = np.linspace(-0.1, 0.2, int(np.prod(shapes[3])), dtype=np.float32).reshape(shapes[3])
            mask_values = np.linspace(0.55, 1.0, int(np.prod(shapes[4])), dtype=np.float32).reshape(shapes[4])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(w_values, dtypes[1])
            inputs_np[2] = from_float32(offset_values, dtypes[2])
            inputs_np[3] = from_float32(bias_values, dtypes[3])
            inputs_np[4] = from_float32(mask_values, dtypes[4])

        if op_name == "attention":
            # Attention 使用有限 4D GQA 样本，覆盖 Q/K/V matmul、causal、softcap 和低精度写回。
            q_values = np.linspace(-1.0, 1.0, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            k_values = np.linspace(0.8, -0.9, int(np.prod(shapes[1])), dtype=np.float32).reshape(shapes[1])
            v_values = np.linspace(-0.6, 0.7, int(np.prod(shapes[2])), dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(q_values, dtypes[0])
            inputs_np[1] = from_float32(k_values, dtypes[1])
            inputs_np[2] = from_float32(v_values, dtypes[2])

        if op_name in {"softmax", "hardmax", "log_softmax"}:
            # Softmax 族算子使用有限样本，覆盖 axis 分段并避免低精度随机 NaN 干扰验证。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-4.0, 4.0, total, dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name in {
            "ceil", "reciprocal", "softplus", "softsign", "hard_sigmoid",
            "elu", "leaky_relu", "selu", "celu", "thresholded_relu", "prelu",
            "hard_swish", "swish", "shrink", "gelu", "mish",
            "round", "erf", "acos", "asin", "cosh", "sinh", "asinh", "acosh", "atanh",
        }:
            # 常见数学/激活类计划使用有限样本，避免极端随机值掩盖主语义和低精度写回路径。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-6.0, 6.0, total, dtype=np.float32).reshape(shapes[0])
            if op_name in {"erf", "cosh", "sinh", "asinh"}:
                values = np.linspace(-2.0, 2.0, total, dtype=np.float32).reshape(shapes[0])
            if op_name in {"acos", "asin"}:
                values = np.linspace(-0.95, 0.95, total, dtype=np.float32).reshape(shapes[0])
            if op_name == "acosh":
                values = np.linspace(1.0, 4.0, total, dtype=np.float32).reshape(shapes[0])
            if op_name == "atanh":
                values = np.linspace(-0.8, 0.8, total, dtype=np.float32).reshape(shapes[0])
            if op_name == "round":
                values = np.linspace(-4.0, 4.0, total, dtype=np.float32).reshape(shapes[0])
                values.reshape(-1)[:8] = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -3.5], dtype=np.float32)
            if op_name == "reciprocal":
                values = np.where(np.abs(values) < 0.5, np.sign(values + 0.01) * 0.5, values)
            if op_name == "hard_swish":
                values.reshape(-1)[:6] = np.array([-4.0, -3.0, -2.0, 0.0, 3.0, 4.0], dtype=np.float32)
            if op_name == "swish":
                values.reshape(-1)[:7] = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0], dtype=np.float32)
            if op_name == "shrink":
                lambd = float(init_args.get("lambd", 0.5))
                values.reshape(-1)[:7] = np.array(
                    [-lambd - 1.0, -lambd, -0.5 * lambd, 0.0, 0.5 * lambd, lambd, lambd + 1.0],
                    dtype=np.float32,
                )
            inputs_np[0] = from_float32(values, dtypes[0])
            if op_name == "prelu":
                slope_total = int(np.prod(shapes[1]))
                slope_values = np.linspace(0.05, 0.65, slope_total, dtype=np.float32).reshape(shapes[1])
                inputs_np[1] = from_float32(slope_values, dtypes[1])

        if op_name == "expand":
            inputs_np[1] = np.array(init_args.get("target_shape", list(shapes[0])), dtype=np.int64)

        if op_name == "reshape":
            inputs_np[1] = np.array(init_args.get("target_shape", [0, -1]), dtype=np.int64)

        if op_name == "tile":
            inputs_np[1] = np.array(init_args.get("repeats_value", [1] * len(shapes[0])), dtype=np.int64)

        if op_name == "concat":
            for idx, (shape, dtype_name) in enumerate(zip(shapes, dtypes)):
                total = int(np.prod(shape))
                values = np.linspace(-3.0 + idx, 3.0 + idx, total, dtype=np.float32).reshape(shape)
                inputs_np[idx] = from_float32(values, dtype_name)

        if op_name == "pad":
            inputs_np[1] = np.array(init_args.get("pads_value", [0] * (2 * len(shapes[0]))), dtype=np.int64)
            const_value = np.array([init_args.get("constant_value", 0.0)], dtype=np.float32)
            inputs_np[2] = from_float32(const_value, dtypes[2])

        if op_name == "center_crop_pad":
            inputs_np[1] = np.array(init_args.get("target_shape", list(shapes[0])), dtype=np.int64)

        if op_name == "slice":
            # Slice 使用固定有限样本与显式 starts/ends/axes/steps，覆盖 C 后端坐标映射和低精度搬运路径。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-3.0, 3.0, total, dtype=np.float32).reshape(shapes[0])
            starts, ends, axes, steps = _slice_io_values(init_args, shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = starts
            inputs_np[2] = ends
            inputs_np[3] = axes
            inputs_np[4] = steps

        if op_name == "compress":
            # Compress 使用显式 bool 条件，避免随机条件导致输出长度和低精度样本不稳定。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-2.5, 2.5, total, dtype=np.float32).reshape(shapes[0])
            condition = np.asarray(
                init_args.get("condition_value", [True] * int(np.prod(shapes[1]))),
                dtype=np.bool_,
            ).reshape(shapes[1])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = condition

        if op_name == "scatter_elements":
            # ScatterElements 构造每条 axis 切片内唯一的目标索引，避免 reduction=none 的重复写入未定义行为。
            data_total = int(np.prod(shapes[0]))
            update_total = int(np.prod(shapes[2]))
            axis = int(init_args.get("axis", 0))
            if axis < 0:
                axis += len(shapes[0])
            dim = int(shapes[0][axis])
            grid = np.indices(shapes[1], dtype=np.int64)
            permutation = (np.arange(dim, dtype=np.int64) + 1) % dim
            indices = permutation[grid[axis]]
            data_values = np.linspace(-2.0, 2.0, data_total, dtype=np.float32).reshape(shapes[0])
            update_values = np.linspace(3.0, -3.0, update_total, dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(data_values, dtypes[0])
            inputs_np[1] = indices
            inputs_np[2] = from_float32(update_values, dtypes[2])

        if op_name == "constant_of_shape":
            inputs_np[0] = np.array(init_args.get("shape_value", list(shapes[0])), dtype=np.int64)

        if op_name == "eye_like":
            values = np.zeros(shapes[0], dtype=np.float32)
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "einsum":
            pass

        if op_name == "topk":
            M, N = shapes[0]
            k_val = init_args.get("k_value", min(4, N))

            # 第二个输入 k
            inputs_np[1] = np.array([k_val], dtype=np.int64)

            # 为了避免 ties，给输入加一点单调扰动
            x = to_float32(inputs_np[0], dtypes[0]).astype(np.float32)
            eps = (np.arange(x.size, dtype=np.float32).reshape(x.shape) * 1e-6)
            inputs_np[0] = from_float32(x + eps, dtypes[0])

        if op_name == "max_unpool":
            inputs_np[1] = np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)

        if op_name == "max_roi_pool":
            inputs_np[1] = np.array(
                [
                    [0.0, 0.0, 0.0, 4.0, 4.0],
                    [1.0, 1.0, 1.0, 3.0, 4.0],
                ],
                dtype=np.float32,
            )
            inputs_np[1] = from_float32(inputs_np[1], dtypes[1])

        if op_name == "roi_align":
            inputs_np[1] = np.array(
                [
                    [0.2, 0.1, 3.8, 3.0],
                    [0.5, 0.4, 4.0, 2.6],
                ],
                dtype=np.float32,
            )
            inputs_np[1] = from_float32(inputs_np[1], dtypes[1])
            inputs_np[2] = np.array([0, 1], dtype=np.int64)

        if op_name == "dft":
            inputs_np[0] = from_float32(np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32), dtypes[0])
            inputs_np[1] = np.array(init_args.get("dft_length_value", shapes[0][1]), dtype=np.int64)

        if op_name == "stft":
            inputs_np[0] = from_float32(np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32), dtypes[0])
            inputs_np[1] = np.array(init_args.get("frame_step_value", 2), dtype=np.int64)
            inputs_np[2] = from_float32(np.array([1.0, 0.5], dtype=np.float32), dtypes[2])
            inputs_np[3] = np.array(init_args.get("frame_length_value", 2), dtype=np.int64)

        if op_name == "rnn":
            inputs_np[0] = from_float32(np.linspace(-0.5, 0.6, 12, dtype=np.float32).reshape(3, 2, 2), dtypes[0])
            inputs_np[1] = from_float32(np.linspace(-0.4, 0.5, 4, dtype=np.float32).reshape(1, 2, 2), dtypes[1])
            inputs_np[2] = from_float32(np.linspace(0.3, -0.2, 4, dtype=np.float32).reshape(1, 2, 2), dtypes[2])
            inputs_np[3] = from_float32(np.linspace(-0.1, 0.2, 4, dtype=np.float32).reshape(1, 4), dtypes[3])
            inputs_np[4] = np.array([3, 2], dtype=np.int64)
            inputs_np[5] = from_float32(np.array([[[0.1, -0.1], [0.2, 0.0]]], dtype=np.float32), dtypes[5])

        if op_name == "gru":
            inputs_np[0] = from_float32(np.linspace(-0.5, 0.6, 12, dtype=np.float32).reshape(3, 2, 2), dtypes[0])
            inputs_np[1] = from_float32(np.linspace(-0.4, 0.5, 12, dtype=np.float32).reshape(1, 6, 2), dtypes[1])
            inputs_np[2] = from_float32(np.linspace(0.3, -0.2, 12, dtype=np.float32).reshape(1, 6, 2), dtypes[2])
            inputs_np[3] = from_float32(np.linspace(-0.2, 0.2, 12, dtype=np.float32).reshape(1, 12), dtypes[3])
            inputs_np[4] = np.array([3, 2], dtype=np.int64)
            inputs_np[5] = from_float32(np.array([[[0.1, -0.1], [0.2, 0.0]]], dtype=np.float32), dtypes[5])

        if op_name == "lstm":
            inputs_np[0] = from_float32(np.linspace(-0.5, 0.6, 12, dtype=np.float32).reshape(3, 2, 2), dtypes[0])
            inputs_np[1] = from_float32(np.linspace(-0.3, 0.4, 16, dtype=np.float32).reshape(1, 8, 2), dtypes[1])
            inputs_np[2] = from_float32(np.linspace(0.2, -0.2, 16, dtype=np.float32).reshape(1, 8, 2), dtypes[2])
            inputs_np[3] = from_float32(np.linspace(-0.1, 0.1, 16, dtype=np.float32).reshape(1, 16), dtypes[3])
            inputs_np[4] = np.array([3, 2], dtype=np.int64)
            inputs_np[5] = from_float32(np.array([[[0.1, 0.0], [-0.1, 0.2]]], dtype=np.float32), dtypes[5])
            inputs_np[6] = from_float32(np.array([[[0.0, 0.2], [0.1, -0.1]]], dtype=np.float32), dtypes[6])
            inputs_np[7] = from_float32(np.linspace(-0.05, 0.05, 6, dtype=np.float32).reshape(1, 6), dtypes[7])

        if op_name == "qlinear_conv":
            out_channels = shapes[3][0]
            inputs_np[0] = np.random.randint(0, 32, size=shapes[0]).astype(np.uint8)
            inputs_np[1] = np.array([0.04], dtype=np.float32)
            inputs_np[2] = np.array([12], dtype=np.uint8)
            inputs_np[3] = np.random.randint(0, 24, size=shapes[3]).astype(np.uint8)
            inputs_np[4] = np.linspace(0.03, 0.06, out_channels, dtype=np.float32)
            inputs_np[5] = np.linspace(7, 11, out_channels, dtype=np.uint8)
            inputs_np[6] = np.array([0.05], dtype=np.float32)
            inputs_np[7] = np.array([121], dtype=np.uint8)

        if op_name == "matmul_integer":
            m, _k = shapes[0]
            _k2, n = shapes[1]
            inputs_np[0] = np.random.randint(0, 32, size=shapes[0]).astype(np.uint8)
            inputs_np[1] = np.random.randint(-16, 16, size=shapes[1]).astype(np.int8)
            inputs_np[2] = np.linspace(3, 9, m, dtype=np.uint8)
            inputs_np[3] = np.linspace(-4, 4, n, dtype=np.int8)

        if op_name == "qlinear_matmul":
            m, _k = shapes[0]
            _k2, n = shapes[3]
            inputs_np[0] = np.random.randint(0, 32, size=shapes[0]).astype(np.uint8)
            inputs_np[1] = np.linspace(0.02, 0.05, m, dtype=np.float32)
            inputs_np[2] = np.linspace(5, 11, m, dtype=np.uint8)
            inputs_np[3] = np.random.randint(0, 24, size=shapes[3]).astype(np.uint8)
            inputs_np[4] = np.linspace(0.03, 0.07, n, dtype=np.float32)
            inputs_np[5] = np.linspace(6, 12, n, dtype=np.uint8)
            inputs_np[6] = np.array([0.04], dtype=np.float32)
            inputs_np[7] = np.array([117], dtype=np.uint8)

        if op_name == "random_uniform_like":
            # 输入只提供 shape，数值本身不会参与 reference 计算
            pass

        inputs_tensor = []
        for data, d in zip(inputs_np, dtypes):
            if data is not None: inputs_tensor.append(Tensor(*data.shape, dtype=d, data=data))
            else: inputs_tensor.append(None)

        # 2. NPS 运行
        try:
            op_init_args = dict(init_args)
            sizes_value = op_init_args.pop("sizes_value", None)
            op_init_args.pop("size_value", None)
            k_value = op_init_args.pop("k_value", None)
            op_init_args.pop("dft_length_value", None)
            op_init_args.pop("frame_step_value", None)
            op_init_args.pop("frame_length_value", None)
            op_init_args.pop("target_shape", None)
            op_init_args.pop("repeats_value", None)
            op_init_args.pop("pads_value", None)
            op_init_args.pop("constant_value", None)
            op_init_args.pop("starts_value", None)
            op_init_args.pop("ends_value", None)
            op_init_args.pop("axes_value", None)
            op_init_args.pop("steps_value", None)
            op_init_args.pop("condition_value", None)
            op_init_args.pop("write_indices_value", None)
            op_init_args.pop("position_ids_value", None)
            op_init_args.pop("image_shape_value", None)
            op_init_args.pop("block_shape_value", None)
            shape_value = op_init_args.pop("shape_value", None)
            fill_value = op_init_args.pop("fill_value", None)
            if op_name == "constant_of_shape" and fill_value is not None:
                op_init_args["value"] = from_float32(np.array([fill_value], dtype=np.float32), out_dtype)

            valid_tensors = [t for t in inputs_tensor if t is not None]

            if op_name == "random_uniform_like":
                low = float(init_args.get("low", 0.0))
                high = float(init_args.get("high", 1.0))
                seed = int(init_args.get("seed", 123))
                nps_out = random_uniform_like_reference(shapes[0], low, high, seed)

            elif op_name in {"conv2d", "conv_transpose", "gemm"}:
                op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **op_init_args)
                nps_out = op.forward(inputs_tensor[0], inputs_tensor[1], inputs_tensor[2])["tensor"].data

            elif op_name == "conv_integer":
                op = op_cls(inputs=[], outputs=[], **op_init_args)
                nps_out = op.forward(inputs_tensor[0], inputs_tensor[1], inputs_tensor[2], inputs_tensor[3])["tensor"].data

            elif op_name in {"rnn", "gru", "lstm"}:
                op = op_cls(inputs=[], outputs=["y"], dtype=out_dtype, **op_init_args)
                nps_out = op.forward(*valid_tensors)["tensor"].data

            elif op_name == "batch_normalization":
                op = op_cls(inputs=[], outputs=["y"], dtype=out_dtype, **op_init_args)
                nps_out = op.forward(*valid_tensors)["tensor"].data

            else:
                op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **op_init_args)

                if op_name in {"cumsum", "cumprod"}:
                    axis_np = np.array([0], dtype=np.int64)
                    axis_tensor = Tensor(*axis_np.shape, dtype="int64", data=axis_np)
                    nps_out = op.forward(valid_tensors[0], axis_tensor)["tensor"].data

                elif op_name == "resize":
                    nps_out = op.forward(valid_tensors[0], valid_tensors[1], valid_tensors[2], valid_tensors[3])["tensor"].data

                elif op_name == "topk":
                    topk_ret = op.forward(valid_tensors[0], valid_tensors[1])["tensor"]
                    nps_out = topk_ret[0].data
                    nps_topk_indices = topk_ret[1].data

                elif op_name == "cast_like":
                    op = op_cls(inputs=[], outputs=[])
                    nps_out = op.forward(valid_tensors[0], valid_tensors[1])["tensor"].data

                else:
                    nps_out = op.forward(*valid_tensors)["tensor"].data

            if op_name in [
                "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
                "reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square",
            ]:
                if np.shape(nps_out) == ():
                    nps_out = np.array([float(nps_out)], dtype=np.float32)
                else:
                    nps_out = np.asarray(nps_out, dtype=np.float32).reshape(1,)

        except Exception as e:
            print(f"  ❌ Iter {i} Crash: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # 3. CUDA 参数打包
        params_bin = None
        if op_name == "conv2d":
            x, w = inputs_np[0], inputs_np[1]
            pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
            oh = (x.shape[2] + pads[0] + pads[2] - d[0]*(w.shape[2]-1) - 1)//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - d[1]*(w.shape[3]-1) - 1)//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[2], w.shape[3],
                      oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "conv_integer":
            x, w, x_zp, w_zp = inputs_np[0], inputs_np[1], inputs_np[2], inputs_np[3]
            pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
            oh = (x.shape[2] + pads[0] + pads[2] - d[0]*(w.shape[2]-1) - 1)//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - d[1]*(w.shape[3]-1) - 1)//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[2], w.shape[3],
                      oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g, x_zp.size, w_zp.size]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "qlinear_conv":
            x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp = inputs_np[:8]
            pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
            oh = (x.shape[2] + pads[0] + pads[2] - d[0]*(w.shape[2]-1) - 1)//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - d[1]*(w.shape[3]-1) - 1)//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[2], w.shape[3],
                      oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g,
                      x_scale.size, x_zp.size, w_scale.size, w_zp.size, y_scale.size, y_zp.size]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "conv_transpose":
            x, w = inputs_np[0], inputs_np[1]
            pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
            output_padding = init_args.get('output_padding', [0, 0])
            effective_h = d[0] * (w.shape[2] - 1) + 1
            effective_w = d[1] * (w.shape[3] - 1) + 1
            oh = s[0] * (x.shape[2] - 1) + output_padding[0] + effective_h - pads[0] - pads[2]
            ow = s[1] * (x.shape[3] - 1) + output_padding[1] + effective_w - pads[1] - pads[3]
            out_c = w.shape[1] * g
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[1], w.shape[2], w.shape[3],
                      out_c, oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "deform_conv":
            x, w, offset = inputs_np[0], inputs_np[1], inputs_np[2]
            pads = list(map(int, init_args.get("pads", [0, 0, 0, 0])))
            s = list(map(int, init_args.get("strides", [1, 1])))
            d = list(map(int, init_args.get("dilations", [1, 1])))
            g = int(init_args.get("group", 1))
            offset_group = int(init_args.get("offset_group", 1))
            has_bias = 1 if inputs_np[3] is not None else 0
            has_mask = 1 if inputs_np[4] is not None else 0
            p_list = [
                x.shape[0],
                x.shape[1],
                x.shape[2],
                x.shape[3],
                w.shape[0],
                w.shape[2],
                w.shape[3],
                offset.shape[2],
                offset.shape[3],
                pads[0],
                pads[1],
                pads[2],
                pads[3],
                s[0],
                s[1],
                d[0],
                d[1],
                g,
                offset_group,
                has_bias,
                has_mask,
            ]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "attention":
            q, k, v = inputs_np[0], inputs_np[1], inputs_np[2]
            mask = inputs_np[3] if len(inputs_np) > 3 else None
            mask_shape = [1, 1, 1, 1]
            mask_rank = 0
            if mask is not None:
                mask_rank = mask.ndim
                mask_shape[:mask_rank] = list(mask.shape)
            int_params = np.array(
                [
                    q.shape[0],
                    q.shape[1],
                    k.shape[1],
                    q.shape[2],
                    k.shape[2],
                    q.shape[3],
                    v.shape[3],
                    1 if mask is not None else 0,
                    1 if len(dtypes) > 3 and dtypes[3] == "bool" else 0,
                    mask_rank,
                    *mask_shape,
                    int(init_args.get("is_causal", 0)),
                ],
                dtype=np.int32,
            )
            float_params = np.array(
                [
                    float(init_args["scale"]) if init_args.get("scale") is not None else -1.0,
                    float(init_args.get("softcap", 0.0) or 0.0),
                ],
                dtype=np.float32,
            )
            params_bin = int_params.tobytes() + float_params.tobytes()
        elif op_name == "matmul_integer":
            a, b, a_zp, b_zp = inputs_np[0], inputs_np[1], inputs_np[2], inputs_np[3]
            M, K = a.shape
            K2, N = b.shape
            assert K == K2
            params_bin = np.array([M, K, N, a_zp.size, b_zp.size], dtype=np.int32).tobytes()
        elif op_name == "qlinear_matmul":
            a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp = inputs_np[:8]
            M, K = a.shape
            K2, N = b.shape
            assert K == K2
            params_bin = np.array(
                [M, K, N, a_scale.size, a_zp.size, b_scale.size, b_zp.size, y_scale.size, y_zp.size],
                dtype=np.int32,
            ).tobytes()
        elif op_name == "max_pool":
            x = inputs_np[0]
            k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
            oh = (x.shape[2] + pads[0] + pads[2] - k[0])//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - k[1])//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow, k[0], k[1], pads[0], pads[1], s[0], s[1]]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "average_pool":
            x = inputs_np[0]
            k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
            d = init_args.get('dilations', [1, 1])
            count_include_pad = init_args.get('count_include_pad', 0)
            kernel_extent_h = d[0] * (k[0] - 1) + 1
            kernel_extent_w = d[1] * (k[1] - 1) + 1
            oh = (x.shape[2] + pads[0] + pads[2] - kernel_extent_h)//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - kernel_extent_w)//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow,
                      k[0], k[1], pads[0], pads[1], s[0], s[1], d[0], d[1], count_include_pad]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "lp_pool":
            x = inputs_np[0]
            k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
            d = init_args.get('dilations', [1, 1])
            p_norm = init_args.get('p', 2)
            kernel_extent_h = d[0] * (k[0] - 1) + 1
            kernel_extent_w = d[1] * (k[1] - 1) + 1
            oh = (x.shape[2] + pads[0] + pads[2] - kernel_extent_h)//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - kernel_extent_w)//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow,
                      k[0], k[1], pads[0], pads[1], s[0], s[1], d[0], d[1], p_norm]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name in {"global_average_pool", "global_max_pool", "global_lp_pool"}:
            x = inputs_np[0]
            spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
            if op_name == "global_lp_pool":
                params_bin = np.array([x.shape[0], x.shape[1], spatial_size, init_args.get('p', 2)], dtype=np.int32).tobytes()
            else:
                params_bin = np.array([x.shape[0], x.shape[1], spatial_size], dtype=np.int32).tobytes()
        elif op_name == "lrn":
            x = inputs_np[0]
            spatial_size = int(np.prod(x.shape[2:]))
            int_params = np.array(
                [x.shape[0], x.shape[1], spatial_size, int(init_args.get("size", 1))],
                dtype=np.int32,
            )
            float_params = np.array(
                [
                    float(init_args.get("alpha", 0.0001)),
                    float(init_args.get("beta", 0.75)),
                    float(init_args.get("bias", 1.0)),
                ],
                dtype=np.float32,
            )
            params_bin = int_params.tobytes() + float_params.tobytes()
        elif op_name == "max_unpool":
            x = inputs_np[0]
            k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
            oh = (x.shape[2] - 1) * s[0] - pads[0] - pads[2] + k[0]
            ow = (x.shape[3] - 1) * s[1] - pads[1] - pads[3] + k[1]
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow,
                      k[0], k[1], pads[0], pads[1], pads[2], pads[3], s[0], s[1]]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "grid_sample":
            x, grid = inputs_np[0], inputs_np[1]
            mode = {"linear": "bilinear", "cubic": "bicubic"}.get(init_args.get("mode", "bilinear"), init_args.get("mode", "bilinear"))
            mode_code = {"bilinear": 0, "nearest": 1, "bicubic": 2}[mode]
            padding_code = {"zeros": 0, "border": 1, "reflection": 2}[init_args.get("padding_mode", "zeros")]
            p_list = [
                x.shape[0],
                x.shape[1],
                x.shape[2],
                x.shape[3],
                grid.shape[1],
                grid.shape[2],
                mode_code,
                padding_code,
                int(init_args.get("align_corners", 0)),
            ]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "max_roi_pool":
            x, rois = inputs_np[0], inputs_np[1]
            pooled_shape = init_args["pooled_shape"]
            ints = np.array(
                [x.shape[0], x.shape[1], x.shape[2], x.shape[3], rois.shape[0], pooled_shape[0], pooled_shape[1]],
                dtype=np.int32,
            ).tobytes()
            params_bin = ints + np.array([init_args.get("spatial_scale", 1.0)], dtype=np.float32).tobytes()
        elif op_name == "roi_align":
            x, rois = inputs_np[0], inputs_np[1]
            mode_code = 0 if init_args.get("mode", "avg").lower() == "avg" else 1
            coord_mode = init_args.get("coordinate_transformation_mode", "half_pixel").lower()
            coord_code = 0 if coord_mode == "half_pixel" else 1
            ints = np.array(
                [
                    x.shape[0],
                    x.shape[1],
                    x.shape[2],
                    x.shape[3],
                    rois.shape[0],
                    init_args["output_height"],
                    init_args["output_width"],
                    init_args.get("sampling_ratio", 0),
                    mode_code,
                    coord_code,
                ],
                dtype=np.int32,
            ).tobytes()
            params_bin = ints + np.array([init_args.get("spatial_scale", 1.0)], dtype=np.float32).tobytes()
        elif op_name == "dft":
            x = inputs_np[0]
            dft_length = int(np.asarray(inputs_np[1]).item())
            axis = init_args.get("axis", 1)
            inverse = init_args.get("inverse", 0)
            onesided = init_args.get("onesided", 0)
            output_axis_len = dft_length // 2 + 1 if onesided and not inverse else dft_length
            output_complex_dim = 1 if inverse and onesided else 2
            params_bin = np.array(
                [
                    x.shape[0],
                    x.shape[1],
                    x.shape[2],
                    output_axis_len,
                    output_complex_dim,
                    axis,
                    inverse,
                    onesided,
                    dft_length,
                ],
                dtype=np.int32,
            ).tobytes()
        elif op_name == "stft":
            signal = inputs_np[0]
            frame_step = int(np.asarray(inputs_np[1]).item())
            frame_length = int(np.asarray(inputs_np[3]).item())
            n_frames = 1 + (signal.shape[1] - frame_length) // frame_step
            bins = frame_length // 2 + 1 if init_args.get("onesided", 1) else frame_length
            has_window = 1 if inputs_np[2] is not None else 0
            params_bin = np.array(
                [
                    signal.shape[0],
                    signal.shape[1],
                    signal.shape[2],
                    n_frames,
                    bins,
                    frame_step,
                    frame_length,
                    init_args.get("onesided", 1),
                    has_window,
                ],
                dtype=np.int32,
            ).tobytes()
        elif op_name in {"rnn", "gru", "lstm"}:
            x = inputs_np[0]
            w = inputs_np[1]
            hidden = int(init_args.get("hidden_size", inputs_np[2].shape[-1]))
            direction_code = {"forward": 0, "reverse": 1, "bidirectional": 2}[init_args.get("direction", "forward")]
            layout = int(init_args.get("layout", 0))
            seq_len = x.shape[1] if layout == 1 else x.shape[0]
            batch = x.shape[0] if layout == 1 else x.shape[1]
            input_size = x.shape[2]
            num_dirs = w.shape[0]
            op_specific = 0
            if op_name == "gru":
                op_specific = int(init_args.get("linear_before_reset", 0))
            elif op_name == "lstm":
                op_specific = int(init_args.get("input_forget", 0))
            params_bin = np.array(
                [seq_len, batch, input_size, num_dirs, hidden, direction_code, layout, op_specific],
                dtype=np.int32,
            ).tobytes()
        elif op_name == "gemm":
            a, b, c = inputs_np[0], inputs_np[1], inputs_np[2]
            tA, tB = init_args['transA'], init_args['transB']
            M = a.shape[0] if tA==0 else a.shape[1]
            K = a.shape[1] if tA==0 else a.shape[0]
            N = b.shape[1] if tB==0 else b.shape[0]
            has_c = 1 if c is not None else 0
            c_type = 0
            if has_c:
                if c.size == 1: c_type=1
                elif c.ndim==1 or (c.ndim==2 and c.shape[0]==1): c_type=2
                elif c.ndim==2 and c.shape[1]==1: c_type=3
                else: c_type=4
            ints = np.array([M, N, K, tA, tB, c_type, has_c], dtype=np.int32).tobytes()
            floats = np.array([init_args['alpha'], init_args['beta']], dtype=np.float32).tobytes()
            params_bin = ints + floats
        elif op_name in {"softmax", "hardmax", "log_softmax"}:
            x, axis = inputs_np[0], init_args['axis']
            if axis < 0: axis += x.ndim
            inner, outer = x.shape[axis], int(np.prod(x.shape[:axis]))
            rem = int(np.prod(x.shape[axis+1:]))
            params_bin = np.array([outer, inner, rem], dtype=np.int32).tobytes()
        elif op_name == "quantize_linear":
            is_signed = 1 if "int8" in out_dtype and "uint8" not in out_dtype else 0
            use_float_math = 0 if "float64" in {dtypes[0], dtypes[1]} else 1
            params_bin = np.array([is_signed, use_float_math], dtype=np.int32).tobytes()
        elif op_name == "matmul":
            M, K = shapes[0]
            K2, N = shapes[1]
            assert K2 == K
            params_bin = np.array([M, K, N], dtype=np.int32).tobytes()
        elif op_name == "reduce_mean":
            M, N = shapes[0]
            params_bin = np.array([M, N], dtype=np.int32).tobytes()

        elif op_name == "gather":
            # 主路径：data=(M,N), indices=(I,), axis=0
            M, N = shapes[0]
            (I,) = shapes[1]
            params_bin = np.array([M, N, I], dtype=np.int32).tobytes()
    
        elif op_name == "scatternd":
            M, N = shapes[0]         # data: (M,N)
            I, K = shapes[1]         # indices: (I,2)
            assert K == 2
            (I2,) = shapes[2]        # updates: (I,)
            assert I2 == I
            params_bin = np.array([M, N, I], dtype=np.int32).tobytes()
        elif op_name == "tensor_scatter":
            cache_shape = list(shapes[0])
            update_shape = list(shapes[1])
            rank = len(cache_shape)
            axis = int(init_args.get("axis", -2))
            if axis < 0:
                axis += rank
            mode_code = 1 if init_args.get("mode", "linear") == "circular" else 0
            params_bin = np.array([rank, axis, mode_code, *cache_shape, *update_shape], dtype=np.int32).tobytes()
        elif op_name in [
            "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
            "reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square",
        ]:
            in_len = int(inputs_np[0].size) 
            params_bin = np.array([in_len], dtype=np.int64).tobytes()

        elif op_name == "gather_elements":
            M, N = shapes[0]
            axis = init_args.get("axis", 1)
            params_bin = np.array([M, N, axis], dtype=np.int32).tobytes()

        elif op_name == "gathernd":
            A, B = shapes[0]
            I, K = shapes[1]
            params_bin = np.array([A, B, I, K], dtype=np.int32).tobytes()

        elif op_name in {"cumsum", "cumprod"}:
            N = int(np.prod(shapes[0]))
            exclusive = int(init_args.get("exclusive", 0))
            reverse = int(init_args.get("reverse", 0))
            params_bin = np.array([N, exclusive, reverse], dtype=np.int32).tobytes()

        elif op_name == "nonzero":
            x = inputs_np[0]
            rank = x.ndim
            dims = np.array(list(x.shape), dtype=np.int32)
            params_bin = np.array([rank], dtype=np.int32).tobytes() + dims.tobytes()

        elif op_name == "argmin" or op_name == "argmax":
            M, N = shapes[0]
            axis = init_args.get("axis", 1)
            keepdims = init_args.get("keepdims", 0)
            select_last_index = init_args.get("select_last_index", 0)
            params_bin = np.array([M, N, axis, keepdims, select_last_index], dtype=np.int32).tobytes()

        elif op_name == "resize":
            N, C, IH, IW = shapes[0]
            OH, OW = init_args["sizes_value"][2], init_args["sizes_value"][3]
            params_bin = np.array([N, C, IH, IW, OH, OW], dtype=np.int32).tobytes()

        elif op_name == "affine_grid":
            size_value = list(map(int, init_args.get("size_value", list(inputs_np[1]))))
            spatial_rank = len(size_value) - 2
            if spatial_rank == 2:
                n, _, h, w = size_value
                d = 1
            elif spatial_rank == 3:
                n, _, d, h, w = size_value
            else:
                raise ValueError(f"AffineGrid size_value must have rank 4 or 5, got {size_value}")
            params_bin = np.array(
                [spatial_rank, n, d, h, w, int(init_args.get("align_corners", 0))],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "expand":
            input_shape = list(shapes[0])
            target_shape = list(map(int, init_args.get("target_shape", input_shape)))
            aligned_input = [1] * (len(target_shape) - len(input_shape)) + input_shape
            output_shape = [
                in_dim if target_dim == -1 else target_dim
                for in_dim, target_dim in zip(aligned_input, target_shape)
            ]
            params_bin = np.array([len(input_shape), len(output_shape), *input_shape, *output_shape], dtype=np.int32).tobytes()

        elif op_name == "reshape":
            params_bin = None

        elif op_name == "squeeze":
            params_bin = None

        elif op_name == "unsqueeze":
            params_bin = None

        elif op_name == "transpose":
            input_shape = list(shapes[0])
            rank = len(input_shape)
            perm = init_args.get("perm")
            if perm is None:
                perm = list(reversed(range(rank)))
            perm = [axis + rank if axis < 0 else axis for axis in perm]
            params_bin = np.array([rank, *input_shape, *perm], dtype=np.int32).tobytes()

        elif op_name == "tile":
            input_shape = list(shapes[0])
            repeats = list(map(int, init_args.get("repeats_value", [1] * len(input_shape))))
            params_bin = np.array([len(input_shape), *input_shape, *repeats], dtype=np.int32).tobytes()

        elif op_name == "concat":
            rank = len(shapes[0])
            axis = init_args.get("axis", 0)
            if axis < 0:
                axis += rank
            shape_values = []
            for shape in shapes:
                shape_values.extend(shape)
            params_bin = np.array([len(shapes), rank, axis, *shape_values], dtype=np.int32).tobytes()

        elif op_name == "pad":
            input_shape = list(shapes[0])
            rank = len(input_shape)
            pads = list(map(int, init_args.get("pads_value", [0] * (2 * rank))))
            output_shape = [input_shape[i] + pads[i] + pads[i + rank] for i in range(rank)]
            mode_code = {"constant": 0, "reflect": 1, "edge": 2, "wrap": 3}.get(init_args.get("mode", "constant"), 0)
            params_bin = np.array([rank, mode_code, *input_shape, *output_shape], dtype=np.int32).tobytes()

        elif op_name == "center_crop_pad":
            input_shape = list(shapes[0])
            output_shape = list(np.asarray(nps_out).shape)
            params_bin = np.array([len(input_shape), *input_shape, *output_shape], dtype=np.int32).tobytes()

        elif op_name == "depth_to_space":
            n, c, h, w = shapes[0]
            mode_code = 0 if init_args.get("mode", "DCR") == "DCR" else 1
            params_bin = np.array(
                [n, c, h, w, int(init_args["blocksize"]), mode_code],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "space_to_depth":
            n, c, h, w = shapes[0]
            params_bin = np.array(
                [n, c, h, w, int(init_args["blocksize"])],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "slice":
            input_shape = list(map(int, shapes[0]))
            output_shape = list(map(int, np.asarray(nps_out).shape))
            starts, ends, axes, steps = _slice_io_values(init_args, input_shape)
            full_starts, _full_ends, full_steps = _normalize_slice_parameters(input_shape, starts, ends, axes, steps)
            params_bin = np.array(
                [len(input_shape), *input_shape, *output_shape, *full_starts, *full_steps],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "compress":
            input_shape = list(map(int, shapes[0]))
            output_shape = list(map(int, np.asarray(nps_out).shape))
            axis_value = init_args.get("axis", None)
            axis = -1
            if axis_value is not None:
                axis = int(axis_value)
                if axis < 0:
                    axis += len(input_shape)
            params_bin = np.array(
                [len(input_shape), len(output_shape), axis, int(np.prod(shapes[1])), *input_shape, *output_shape],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "scatter_elements":
            data_shape = list(map(int, shapes[0]))
            update_shape = list(map(int, shapes[2]))
            axis = int(init_args.get("axis", 0))
            if axis < 0:
                axis += len(data_shape)
            reduction = {"none": 0, "add": 1, "mul": 2}.get(init_args.get("reduction", "none"), 0)
            params_bin = np.array(
                [len(data_shape), axis, reduction, *data_shape, *update_shape],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "constant_of_shape":
            target_shape = list(map(int, init_args.get("shape_value", list(shapes[0]))))
            fill_value = float(init_args.get("fill_value", 0.0))
            params_bin = (
                np.array([len(target_shape), *target_shape], dtype=np.int32).tobytes()
                + np.array([fill_value], dtype=np.float32).tobytes()
            )

        elif op_name == "eye_like":
            rows, cols = shapes[0]
            params_bin = np.array([rows, cols, int(init_args.get("k", 0))], dtype=np.int32).tobytes()

        elif op_name in {"cast", "cast_like"}:
            output_kind = {"float": 0, "int32": 1, "int64": 2, "bool": 3}.get(out_dtype, 0)
            params_bin = np.array([output_kind], dtype=np.int32).tobytes()

        elif op_name == "bitcast":
            params_bin = np.array([np.dtype(nn.DTYPE_TO_NUMPY[out_dtype]).itemsize], dtype=np.int32).tobytes()

        elif op_name == "bit_shift":
            direction = 0 if init_args.get("direction", "LEFT").upper() == "LEFT" else 1
            params_bin = np.array([direction], dtype=np.int32).tobytes()

        elif op_name == "isinf":
            params_bin = np.array(
                [int(init_args.get("detect_positive", 1)), int(init_args.get("detect_negative", 1))],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "size":
            params_bin = np.array([int(np.prod(inputs_np[0].shape, dtype=np.int64))], dtype=np.int64).tobytes()

        elif op_name == "hard_sigmoid":
            params_bin = np.array(
                [float(init_args.get("alpha", 0.2)), float(init_args.get("beta", 0.5))],
                dtype=np.float32,
            ).tobytes()

        elif op_name == "rms_normalization":
            input_shape = list(shapes[0])
            axis = int(init_args.get("axis", -1))
            if axis < 0:
                axis += len(input_shape)
            normalized_size = int(np.prod(input_shape[axis:]))
            row_count = int(np.prod(input_shape[:axis])) if axis > 0 else 1
            params_bin = (
                np.array([row_count, normalized_size], dtype=np.int32).tobytes()
                + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
            )

        elif op_name == "mean_variance_normalization":
            x = inputs_np[0]
            rank = x.ndim
            axes = init_args.get("axes", [0, 2, 3])
            resolved_axes = []
            for ax in axes:
                axis = int(ax)
                if axis < 0:
                    axis += rank
                resolved_axes.append(axis)
            resolved_axes = sorted(set(resolved_axes))
            params_bin = np.array(
                [rank, len(resolved_axes), *x.shape, *resolved_axes],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "batch_normalization":
            x = inputs_np[0]
            spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
            params_bin = (
                np.array([x.shape[0], x.shape[1], spatial_size], dtype=np.int32).tobytes()
                + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
            )

        elif op_name == "instance_normalization":
            x = inputs_np[0]
            spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
            params_bin = (
                np.array([x.shape[0], x.shape[1], spatial_size], dtype=np.int32).tobytes()
                + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
            )

        elif op_name == "layer_normalization":
            x = inputs_np[0]
            input_shape = list(x.shape)
            axis = int(init_args.get("axis", -1))
            if axis < 0:
                axis += len(input_shape)
            normalized_size = int(np.prod(input_shape[axis:]))
            row_count = int(np.prod(input_shape[:axis])) if axis > 0 else 1
            has_scale = 1 if inputs_np[1] is not None else 0
            has_bias = 1 if inputs_np[2] is not None else 0
            params_bin = (
                np.array([row_count, normalized_size, has_scale, has_bias], dtype=np.int32).tobytes()
                + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
            )

        elif op_name == "lp_normalization":
            x = inputs_np[0]
            axis = int(init_args.get("axis", -1))
            if axis < 0:
                axis += x.ndim
            outer = int(np.prod(x.shape[:axis])) if axis > 0 else 1
            inner = int(x.shape[axis])
            remaining = int(np.prod(x.shape[axis + 1:])) if axis + 1 < x.ndim else 1
            params_bin = np.array([outer, inner, remaining, int(init_args.get("p", 2))], dtype=np.int32).tobytes()

        elif op_name == "group_normalization":
            x = inputs_np[0]
            spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
            params_bin = (
                np.array([x.shape[0], x.shape[1], spatial_size, int(init_args["num_groups"])], dtype=np.int32).tobytes()
                + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
            )

        elif op_name == "rotary_embedding":
            x = inputs_np[0]
            rank = x.ndim
            if rank == 4:
                batch, heads, sequence, head_size = x.shape
            else:
                batch, sequence, hidden = x.shape
                heads = int(init_args["num_heads"])
                head_size = hidden // heads
            rotary_dim = int(init_args.get("rotary_embedding_dim", 0) or head_size)
            has_position_ids = 1 if inputs_np[3] is not None else 0
            cos_rank = inputs_np[1].ndim
            params_bin = np.array(
                [
                    rank,
                    batch,
                    heads,
                    sequence,
                    head_size,
                    rotary_dim,
                    int(init_args.get("interleaved", 0)),
                    has_position_ids,
                    cos_rank,
                ],
                dtype=np.int32,
            ).tobytes()

        elif op_name == "col2im":
            x = inputs_np[0]
            image_shape = [int(v) for v in np.asarray(inputs_np[1], dtype=np.int64).reshape(-1)]
            block_shape = [int(v) for v in np.asarray(inputs_np[2], dtype=np.int64).reshape(-1)]
            spatial_rank = len(image_shape)
            pads = list(map(int, init_args.get("pads", [0] * (2 * spatial_rank))))
            strides = list(map(int, init_args.get("strides", [1] * spatial_rank)))
            dilations = list(map(int, init_args.get("dilations", [1] * spatial_rank)))
            n_blocks = []
            for axis in range(spatial_rank):
                count = (
                    image_shape[axis]
                    + pads[axis]
                    + pads[axis + spatial_rank]
                    - dilations[axis] * (block_shape[axis] - 1)
                    - 1
                ) // strides[axis] + 1
                n_blocks.append(int(count))
            kernel_size = int(np.prod(block_shape))
            block_count = int(np.prod(n_blocks))
            channels = int(x.shape[1] // kernel_size)
            params_bin = np.array(
                [
                    x.shape[0],
                    channels,
                    spatial_rank,
                    x.shape[2],
                    kernel_size,
                    block_count,
                    *image_shape,
                    *block_shape,
                    *pads,
                    *strides,
                    *dilations,
                    *n_blocks,
                ],
                dtype=np.int32,
            ).tobytes()

        elif op_name in {"elu", "leaky_relu", "celu", "thresholded_relu"}:
            params_bin = np.array([float(init_args.get("alpha", 1.0))], dtype=np.float32).tobytes()

        elif op_name == "swish":
            params_bin = np.array([float(init_args.get("alpha", 1.0))], dtype=np.float32).tobytes()

        elif op_name == "selu":
            params_bin = np.array(
                [float(init_args.get("alpha", 1.67326)), float(init_args.get("gamma", 1.0507))],
                dtype=np.float32,
            ).tobytes()

        elif op_name == "gelu" and init_args.get("approximate", "none") == "tanh":
            params_bin = np.array([1], dtype=np.int32).tobytes()

        elif op_name == "shrink":
            params_bin = np.array(
                [float(init_args.get("bias", 0.0)), float(init_args.get("lambd", 0.5))],
                dtype=np.float32,
            ).tobytes()

        elif op_name == "einsum":
            M, K = shapes[0]
            K2, N = shapes[1]
            assert K == K2
            params_bin = np.array([M, K, N], dtype=np.int32).tobytes()

        elif op_name == "topk":
            M, N = shapes[0]
            k_val = int(inputs_np[1].reshape(-1)[0])
            axis = init_args.get("axis", 1)
            largest = init_args.get("largest", 1)
            sorted_flag = init_args.get("sorted", 1)
            params_bin = np.array([M, N, axis, k_val, largest, sorted_flag], dtype=np.int32).tobytes()

        elif op_name == "random_uniform_like":
            numel = int(np.prod(shapes[0]))
            low = float(init_args.get("low", 0.0))
            high = float(init_args.get("high", 1.0))
            seed = np.uint32(int(init_args.get("seed", 123)))
            params_bin = (np.array([numel], dtype=np.int32).tobytes() + np.array([low, high], dtype=np.float32).tobytes() + np.array([seed], dtype=np.uint32).tobytes())

        # 4. 数据转换与 广播处理
        expected_shape = nps_out.shape
        if expected_shape == ():
            expected_shape = (1,) # 统一当成 1 元素张量来跑 CUDA/读写 bin
            nps_out = np.array([nps_out], dtype=nps_out.dtype)
        is_complex_kernel = op_name in ["conv2d", "conv_integer", "qlinear_conv", "conv_transpose", "col2im", "deform_conv", "attention", "matmul_integer", "qlinear_matmul", "max_pool", "average_pool", "lp_pool", "global_average_pool", "global_max_pool", "global_lp_pool", "lrn", "mean_variance_normalization", "batch_normalization", "instance_normalization", "layer_normalization", "lp_normalization", "group_normalization", "max_unpool", "grid_sample", "max_roi_pool", "roi_align", "dft", "stft", "rnn", "gru", "lstm", "gemm", "softmax", "hardmax", "log_softmax"] # 这些算子自己处理形状
        is_double_kernel = is_complex_kernel or op_name in ["quantize_linear", "dequantize_linear"]
        
        cuda_inputs = []
        for i, (inp, d) in enumerate(zip(inputs_np, dtypes)):
            if inp is None:
                cuda_inputs.append(None)
            else:
                if op_name == "bitcast":
                    cuda_inputs.append(np.ascontiguousarray(inp.astype(nn.DTYPE_TO_NUMPY[d], copy=False)))
                    continue

                if op_name in {"bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "bit_shift"} and d == "int32":
                    cuda_inputs.append(np.ascontiguousarray(inp.astype(np.int32, copy=False)))
                    continue

                if  op_name in ["gather", "scatternd", "tensor_scatter", "scatter_elements", "gather_elements", "gathernd","resize", "affine_grid", "topk", "max_unpool", "roi_align", "col2im", "dft", "stft", "rnn", "gru", "lstm", "tile", "expand", "pad", "center_crop_pad", "slice", "constant_of_shape", "rotary_embedding"] and d == "int64":
                    cuda_inputs.append(np.ascontiguousarray(inp.astype(np.int64)))
                    continue

                target_dtype = np.float64 if is_double_kernel else np.float32
                val_f32 = to_float32(inp, d)
                
                # 广播逻辑
                if (not is_complex_kernel) and (op_name not in ["matmul", "reduce_mean","reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square","gather", "gather_elements", "gathernd","scatternd", "tensor_scatter", "scatter_elements", "nonzero", "argmin", "argmax", "size", "resize", "affine_grid", "grid_sample", "einsum", "topk", "random_uniform_like", "expand", "flatten", "reshape", "squeeze", "unsqueeze", "transpose", "tile", "concat", "pad", "center_crop_pad", "depth_to_space", "space_to_depth", "slice", "compress", "constant_of_shape", "eye_like", "rotary_embedding", "col2im", "deform_conv", "attention"]):
                    try:
                        if val_f32.shape != expected_shape:
                            val_f32 = np.broadcast_to(val_f32, expected_shape)
                    except Exception as e:
                        print(f"Warning: Broadcast failed for input {i} in {op_name}: {e}")
                
                cuda_inputs.append(val_f32.astype(target_dtype))
        
        # 5. 执行 CUDA
        # cuda_out = run_cuda_ground_truth(
        #     op_name, 
        #     cuda_inputs, 
        #     params_binary=params_bin, 
        #     output_dtype=np.float64 if is_double_kernel else np.float32,
        #     target_shape=expected_shape
        # ) 
            
        # if cuda_out is None: continue
        
        # out_np_dtype = (np.uint8 if out_dtype == "bool" else (np.float64 if is_double_kernel else np.float32))
        if out_dtype == "bool":
            out_np_dtype = np.uint8
        elif op_name in {"qlinear_conv", "qlinear_matmul"} and out_dtype == "uint8":
            out_np_dtype = np.uint8
        elif out_dtype == "int32":
            out_np_dtype = np.int32
        elif out_dtype == "int64":
            out_np_dtype = np.int64
        elif op_name == "bitcast":
            out_np_dtype = nn.DTYPE_TO_NUMPY[out_dtype]
        else:
            out_np_dtype = np.float64 if is_double_kernel else np.float32

        cuda_out = run_cuda_ground_truth(
        op_name,
        cuda_inputs,
        params_binary=params_bin,
        output_dtype=out_np_dtype,
        target_shape=expected_shape
        )

        if cuda_out is None:
            continue

        if op_name == "topk":
            idx_path = "tmp_out_idx.bin"
            if not os.path.exists(idx_path):
                print(f"  ❌ Iter {i} FAILED")
                print("     Missing tmp_out_idx.bin for TopK")
                break

            cuda_topk_indices = np.fromfile(idx_path, dtype=np.int64).reshape(expected_shape)
            os.remove(idx_path)

            nps_vals = to_float32(nps_out, out_dtype)
            ok_vals, max_abs, max_rel, fail_mask = check_accuracy(nps_vals, cuda_out, atol, rtol, out_dtype)

            ok_idx = np.array_equal(
                np.asarray(nps_topk_indices).astype(np.int64),
                np.asarray(cuda_topk_indices).astype(np.int64)
            )

            if max_abs >= 0:
                stats_abs.append(max_abs)
                stats_rel.append(max_rel)

            if ok_vals and ok_idx:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                if not ok_idx:
                    print("     TopK indices mismatch")
                else:
                    print(f"     Max Abs Diff: {max_abs:.6f} (Limit: {atol})")
                    print(f"     Max Rel Diff: {max_rel:.6f} (Limit: {rtol})")
                break
            continue

        if out_dtype == "bool":
            cuda_out = cuda_out.astype(np.float32)   

        # 6. 对比
        # nps_f32 = to_float32(nps_out, out_dtype)
        # is_ok, max_abs, max_rel, fail_mask = check_accuracy(nps_f32, cuda_out, atol, rtol, out_dtype)
        if op_name == "bitcast":
            nps_raw = np.ascontiguousarray(nps_out).view(np.uint8)
            cuda_raw = np.ascontiguousarray(cuda_out).view(np.uint8)
            is_ok = np.array_equal(nps_raw, cuda_raw)
            max_abs = 0.0 if is_ok else -1.0
            max_rel = 0.0 if is_ok else -1.0
            fail_mask = None if is_ok else (nps_raw != cuda_raw)
            nps_f32 = np.asarray(nps_out).reshape(expected_shape)
        elif out_dtype in {"int32", "int64"}:
            int_dtype = np.int32 if out_dtype == "int32" else np.int64
            nps_int = np.asarray(nps_out).astype(int_dtype)
            cuda_int = np.asarray(cuda_out).astype(int_dtype)
            is_ok = np.array_equal(nps_int, cuda_int)
            max_abs = 0.0 if is_ok else -1.0
            max_rel = 0.0 if is_ok else -1.0
            fail_mask = None if is_ok else (nps_int != cuda_int)
            nps_f32 = nps_int.astype(np.float32)
        else:
            nps_f32 = to_float32(nps_out, out_dtype)
            cuda_out = quantize_to_dtype_float32(cuda_out, out_dtype)
            is_ok, max_abs, max_rel, fail_mask = check_accuracy(nps_f32, cuda_out, atol, rtol, out_dtype)
        
        if max_abs >= 0:
            stats_abs.append(max_abs)
            stats_rel.append(max_rel)
        if is_ok:
            pass_cnt += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            if max_abs == -999.0: print(f"     Failed due to Overflow/Inf Logic Mismatch")
            elif max_abs == -1.0: print(f"     Failed due to NaN/Inf Mismatch")
            else:
                print(f"     Max Abs Diff: {max_abs:.6f} (Limit: {atol})")
                print(f"     Max Rel Diff: {max_rel:.6f} (Limit: {rtol})")
            
            if fail_mask is not None and np.any(fail_mask):
                idx_flat = np.argmax(fail_mask)
                idx = np.unravel_index(idx_flat, fail_mask.shape)
                print(f"     🔍 Debug Sample at {idx}:")
                print(f"        GT (CUDA) = {cuda_out[idx]}")
                print(f"        NPS (C)   = {nps_f32[idx]}")
                # 显示原始输入值
                for k, inp_arr in enumerate(inputs_np):
                    val_disp = ""
                    if inp_arr is None: val_disp = "None"
                    else:
                        try:
                            if (not is_complex_kernel) and (op_name not in ["matmul", "reduce_mean", "gather", "scatternd", "tensor_scatter", "scatter_elements","nonzero", "argmin", "argmax", "size", "resize", "affine_grid", "grid_sample", "einsum", "topk", "random_uniform_like", "expand", "flatten", "reshape", "squeeze", "unsqueeze", "transpose", "tile", "concat", "pad", "center_crop_pad", "depth_to_space", "space_to_depth", "slice", "compress", "constant_of_shape", "eye_like", "rotary_embedding", "col2im", "deform_conv", "attention"]):
                                val_disp = np.broadcast_to(inp_arr, expected_shape)[idx]
                            else:
                                if inp_arr.shape == expected_shape:
                                    val_disp = inp_arr[idx]
                                else:
                                    val_disp = f"Shape{inp_arr.shape} (No direct mapping)"
                        except Exception as e:
                            val_disp = f"Error: {e}"
                            
                    print(f"        Input {k}   = {val_disp}")
            break

    if pass_cnt == iterations:
        print(f"  ✅ Pass ({pass_cnt}/{iterations})\n")
    else:
        print(f"  ⚠️  Fail\n")
    return stats_abs, stats_rel, pass_cnt == iterations
