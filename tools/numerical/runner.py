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
from .data import generate_random_data
from .dtype import from_float32, quantize_to_dtype_float32, to_float32
from .runner_cuda_params import build_cuda_params
from .runner_params import slice_io_values
from .runner_nps import run_nps_forward


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
            if init_args.get("input_values") is not None:
                if dtypes[0] in {"bool", "int2", "uint2", "int4", "uint4", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}:
                    inputs_np[0] = np.asarray(init_args["input_values"], dtype=nn.DTYPE_TO_NUMPY[dtypes[0]]).reshape(shapes[0])
                else:
                    input_values = np.asarray(init_args["input_values"], dtype=np.float32).reshape(shapes[0])
                    inputs_np[0] = from_float32(input_values, dtypes[0])
            if inputs_np[1] is not None:
                if init_args.get("scale_values") is not None:
                    scale_values = np.asarray(init_args["scale_values"], dtype=np.float32).reshape(shapes[1])
                    inputs_np[1] = from_float32(scale_values, dtypes[1])
                else:
                    inputs_np[1] = np.abs(inputs_np[1]) + 1e-4
            if init_args.get("omit_zero_point") and len(inputs_np) > 2:
                zp_dtype = nn.DTYPE_TO_NUMPY[dtypes[2]]
                inputs_np[2] = np.zeros(shapes[2], dtype=zp_dtype)
            elif len(inputs_np) > 2 and inputs_np[2] is not None:
                if init_args.get("zero_point_values") is not None:
                    if dtypes[2] in {"float4_e2m1", "float8_e8m0"}:
                        zp_values = np.asarray(init_args["zero_point_values"], dtype=np.float32).reshape(shapes[2])
                        inputs_np[2] = from_float32(zp_values, dtypes[2])
                    else:
                        zp_dtype = nn.DTYPE_TO_NUMPY[dtypes[2]]
                        inputs_np[2] = np.asarray(init_args["zero_point_values"], dtype=zp_dtype).reshape(shapes[2])
                else:
                    inputs_np[2] = np.round(inputs_np[2])
                    if op_name == "quantize_linear":
                        if dtypes[2] == "uint2":
                            inputs_np[2] = np.clip(inputs_np[2], 0, 3)
                        elif dtypes[2] == "int2":
                            inputs_np[2] = np.clip(inputs_np[2], -2, 1)
                        elif dtypes[2] == "uint4":
                            inputs_np[2] = np.clip(inputs_np[2], 0, 15)
                        elif dtypes[2] == "int4":
                            inputs_np[2] = np.clip(inputs_np[2], -8, 7)
                        elif dtypes[2] == "uint8":
                            inputs_np[2] = np.clip(inputs_np[2], 0, 255)
                        else:
                            inputs_np[2] = np.clip(inputs_np[2], -128, 127)
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
            # GridSample 使用固定有限样本，按计划覆盖不同 mode/padding/align_corners 属性组合。
            x_values = np.linspace(-1.5, 1.5, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            variant = init_args.get("grid_variant", "linear_reflection")
            if variant == "nearest_border":
                grid_values = np.array(
                    [
                        [
                            [[-1.40, -1.30], [-0.60, -0.50], [-0.20, 0.00], [1.30, 1.20]],
                            [[0.20, 0.50], [0.60, 1.40], [-1.00, 1.00], [1.00, -1.00]],
                            [[-0.35, 0.25], [0.45, -0.75], [1.15, 0.15], [-1.15, 0.85]],
                        ]
                    ],
                    dtype=np.float32,
                )
            elif variant == "cubic_zeros":
                grid_values = np.array(
                    [
                        [
                            [[-1.25, -1.15], [-0.75, -0.35], [-0.10, 0.20], [1.20, -0.65]],
                            [[-0.55, 0.55], [0.35, -0.15], [0.80, 0.75], [1.35, 1.15]],
                            [[-1.10, 1.05], [-0.25, 0.95], [0.55, 0.35], [1.05, -1.05]],
                        ]
                    ],
                    dtype=np.float32,
                )
            else:
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

        if op_name in {"tril", "triu", "trilu"}:
            # 三角矩阵类算子使用固定二维样本和显式 k，覆盖上下三角遮罩与低精度位模式搬运。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-3.5, 4.5, total, dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = np.asarray(init_args.get("k_value", -1), dtype=np.int64)

        if op_name == "range":
            # Range 使用固定标量，覆盖正 delta 与非整数步长，避免随机 delta 为 0 或输出过长。
            inputs_np[0] = from_float32(np.asarray(init_args.get("start_value", -2.0), dtype=np.float32), dtypes[0])
            inputs_np[1] = from_float32(np.asarray(init_args.get("limit_value", 3.0), dtype=np.float32), dtypes[1])
            inputs_np[2] = from_float32(np.asarray(init_args.get("delta_value", 0.75), dtype=np.float32), dtypes[2])

        if op_name == "one_hot":
            # OneHot 使用正负混合索引，覆盖负索引归一化、越界忽略和 axis 插入语义。
            indices = np.asarray([[0, 1, -1], [3, 4, -5]], dtype=np.int64).reshape(shapes[0])
            values = np.asarray(init_args.get("values_value", [-0.5, 2.0]), dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = indices
            inputs_np[1] = np.asarray(init_args.get("depth_value", 4), dtype=np.int64)
            inputs_np[2] = from_float32(values, dtypes[2])

        if op_name == "reverse_sequence":
            # ReverseSequence 使用每个 batch 不同的 sequence length，覆盖部分反转和保持尾部不变。
            total = int(np.prod(shapes[0]))
            values = np.linspace(-2.0, 2.0, total, dtype=np.float32).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = np.asarray(init_args.get("sequence_lens_value", [shapes[0][0]] * shapes[0][1]), dtype=np.int64)

        if op_name == "det":
            # Det 使用两组非奇异 3x3 矩阵，覆盖 batch determinant 和低精度输入量化后计算。
            values = np.asarray(
                [
                    [[2.0, -1.0, 0.5], [1.0, 3.0, -2.0], [0.0, 1.5, 4.0]],
                    [[-1.0, 2.0, 1.0], [0.5, -3.0, 2.5], [3.0, 0.0, 1.0]],
                ],
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "mel_weight_matrix":
            # MelWeightMatrix 的五个输入都是标量，使用固定语音频段参数覆盖三角滤波器生成。
            inputs_np[0] = np.asarray(init_args.get("num_mel_bins_value", 4), dtype=np.int64)
            inputs_np[1] = np.asarray(init_args.get("dft_length_value", 10), dtype=np.int64)
            inputs_np[2] = np.asarray(init_args.get("sample_rate_value", 16000), dtype=np.int64)
            inputs_np[3] = np.asarray(init_args.get("lower_edge_hertz_value", 20.0), dtype=np.float32)
            inputs_np[4] = np.asarray(init_args.get("upper_edge_hertz_value", 7600.0), dtype=np.float32)

        if op_name in {"hann_window", "hamming_window", "blackman_window"}:
            # Window 算子输入是标量 size；显式设置可避免随机整数生成非法窗口长度。
            inputs_np[0] = np.asarray(init_args.get("window_size_value", 8), dtype=np.int64)

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
            # BatchNormalization 使用固定有限样本，覆盖推理/训练公式、通道参数广播和低精度写回主路径。
            x_values = np.linspace(-1.2, 1.3, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            channel_count = int(shapes[1][0])
            scale_values = np.linspace(0.75, 1.5, channel_count, dtype=np.float32).reshape(shapes[1])
            bias_values = np.linspace(-0.2, 0.3, channel_count, dtype=np.float32).reshape(shapes[2])
            mean_values = np.linspace(-0.1, 0.2, channel_count, dtype=np.float32).reshape(shapes[3])
            var_values = np.linspace(0.5, 2.0, channel_count, dtype=np.float32).reshape(shapes[4])
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
            # LayerNormalization 使用后缀维度 scale/bias，覆盖 C 后端承载的单输出主路径和低精度写回。
            x_values = np.linspace(-1.8, 1.4, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            scale_values = np.linspace(0.5, 1.7, int(np.prod(shapes[1])), dtype=np.float32).reshape(shapes[1])
            bias_values = np.linspace(-0.3, 0.3, int(np.prod(shapes[2])), dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(scale_values, dtypes[1])
            inputs_np[2] = from_float32(bias_values, dtypes[2])

        if op_name == "lp_normalization":
            # LpNormalization 使用固定有限样本，覆盖 p=1/p=2 下按 axis 归一化且保留输入符号的路径。
            if "input_values" in init_args:
                values = np.asarray(init_args["input_values"], dtype=np.float32).reshape(shapes[0])
            else:
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
            # DeformConv 使用小幅 offset 和有限 mask，覆盖双线性采样、可选 bias/mask、分组和低精度写回路径。
            x_values = np.linspace(-1.2, 1.2, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            w_values = np.linspace(-0.7, 0.8, int(np.prod(shapes[1])), dtype=np.float32).reshape(shapes[1])
            offset_values = np.linspace(-0.25, 0.25, int(np.prod(shapes[2])), dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(w_values, dtypes[1])
            inputs_np[2] = from_float32(offset_values, dtypes[2])
            if len(inputs_np) > 3 and inputs_np[3] is not None:
                bias_values = np.linspace(-0.1, 0.2, int(np.prod(shapes[3])), dtype=np.float32).reshape(shapes[3])
                inputs_np[3] = from_float32(bias_values, dtypes[3])
            if len(inputs_np) > 4 and inputs_np[4] is not None:
                mask_values = np.linspace(0.55, 1.0, int(np.prod(shapes[4])), dtype=np.float32).reshape(shapes[4])
                inputs_np[4] = from_float32(mask_values, dtypes[4])

        if op_name == "attention":
            # Attention 使用有限 4D GQA 样本，覆盖 Q/K/V matmul、mask、causal、softcap 和低精度写回。
            q_values = np.linspace(-1.0, 1.0, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            k_values = np.linspace(0.8, -0.9, int(np.prod(shapes[1])), dtype=np.float32).reshape(shapes[1])
            v_values = np.linspace(-0.6, 0.7, int(np.prod(shapes[2])), dtype=np.float32).reshape(shapes[2])
            inputs_np[0] = from_float32(q_values, dtypes[0])
            inputs_np[1] = from_float32(k_values, dtypes[1])
            inputs_np[2] = from_float32(v_values, dtypes[2])
            if len(inputs_np) > 3 and inputs_np[3] is not None:
                mask_shape = tuple(shapes[3])
                mask_variant = init_args.get("attention_mask_variant", "float_bias")
                if dtypes[3] == "bool":
                    mask_values = np.ones(mask_shape, dtype=np.bool_)
                    if mask_values.size:
                        flat = mask_values.reshape(-1)
                        flat[-1] = False
                        if mask_variant == "bool_broadcast" and flat.size > 2:
                            flat[1] = False
                    inputs_np[3] = mask_values
                else:
                    mask_values = np.zeros(mask_shape, dtype=np.float32)
                    if mask_values.size:
                        flat = mask_values.reshape(-1)
                        flat[min(1, flat.size - 1)] = -1.0e4
                        if mask_variant == "float_bias" and flat.size > 3:
                            flat[-1] = -0.75
                    inputs_np[3] = from_float32(mask_values, dtypes[3])

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
            starts, ends, axes, steps = slice_io_values(init_args, shapes[0])
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
            if init_args.get("roi_variant") == "scaled_clipped":
                inputs_np[1] = np.array(
                    [
                        [0.0, -2.0, -2.0, 9.0, 11.0],
                        [1.0, 2.0, 1.0, 12.0, 9.0],
                        [0.0, 30.0, 30.0, 32.0, 32.0],
                    ],
                    dtype=np.float32,
                )
            else:
                inputs_np[1] = np.array(
                    [
                        [0.0, 0.0, 0.0, 4.0, 4.0],
                        [1.0, 1.0, 1.0, 3.0, 4.0],
                    ],
                    dtype=np.float32,
                )
            inputs_np[1] = from_float32(inputs_np[1], dtypes[1])

        if op_name == "roi_align":
            if init_args.get("roi_variant") == "max_output_half_pixel":
                inputs_np[1] = np.array(
                    [
                        [-0.5, -0.25, 4.4, 4.2],
                        [0.75, 0.5, 6.6, 5.6],
                        [4.8, 3.9, 5.6, 4.5],
                    ],
                    dtype=np.float32,
                )
                inputs_np[2] = np.array([0, 1, 0], dtype=np.int64)
            else:
                inputs_np[1] = np.array(
                    [
                        [0.2, 0.1, 3.8, 3.0],
                        [0.5, 0.4, 4.0, 2.6],
                    ],
                    dtype=np.float32,
                )
                inputs_np[2] = np.array([0, 1], dtype=np.int64)
            inputs_np[1] = from_float32(inputs_np[1], dtypes[1])

        if op_name == "dft":
            dft_variant = init_args.get("dft_variant", "real_onesided")
            if dft_variant == "complex_full":
                values = np.array(
                    [[[1.0, 0.25], [2.0, -0.5], [-1.0, 0.75], [0.5, -1.25]]],
                    dtype=np.float32,
                )
            elif dft_variant == "inverse_onesided":
                values = np.array([[[10.0, 0.0], [-2.0, 1.0], [3.0, 0.0]]], dtype=np.float32)
            elif dft_variant == "high_rank_axis":
                values = np.linspace(-2.0, 2.5, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
            else:
                values = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = np.array(init_args.get("dft_length_value", shapes[0][1]), dtype=np.int64)

        if op_name == "stft":
            stft_variant = init_args.get("stft_variant", "windowed_onesided")
            if stft_variant == "complex_no_window_full":
                signal_values = np.array(
                    [[[1.0, 0.25], [2.0, -0.5], [3.0, 0.75], [4.0, -1.25], [5.0, 0.5]]],
                    dtype=np.float32,
                )
                window_values = None
            elif stft_variant == "real_window_full":
                signal_values = np.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=np.float32)
                window_values = np.array([1.0, 0.5, 0.25], dtype=np.float32)
            elif stft_variant == "high_rank_prefix":
                signal_values = np.linspace(-1.5, 2.0, int(np.prod(shapes[0])), dtype=np.float32).reshape(shapes[0])
                window_values = np.array([1.0, 0.5, -0.25, 0.75], dtype=np.float32)
            else:
                signal_values = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
                window_values = np.array([1.0, 0.5], dtype=np.float32)
            inputs_np[0] = from_float32(signal_values, dtypes[0])
            inputs_np[1] = np.array(init_args.get("frame_step_value", 2), dtype=np.int64)
            inputs_np[2] = None if window_values is None else from_float32(window_values, dtypes[2])
            inputs_np[3] = np.array(init_args.get("frame_length_value", 2), dtype=np.int64)

        if op_name in {"rnn", "gru", "lstm"}:
            layout = int(init_args.get("layout", 0))
            direction = init_args.get("direction", "forward")
            num_dirs = 2 if direction == "bidirectional" else 1
            x_shape = tuple(shapes[0])
            seq_len = x_shape[1] if layout == 1 else x_shape[0]
            batch = x_shape[0] if layout == 1 else x_shape[1]
            input_size = x_shape[2]
            hidden = int(init_args.get("hidden_size", shapes[2][-1]))
            gates = {"rnn": 1, "gru": 3, "lstm": 4}[op_name]
            x_values = np.linspace(-0.5, 0.6, seq_len * batch * input_size, dtype=np.float32).reshape(x_shape)
            inputs_np[0] = from_float32(x_values, dtypes[0])
            inputs_np[1] = from_float32(
                np.linspace(-0.4, 0.5, num_dirs * gates * hidden * input_size, dtype=np.float32).reshape(num_dirs, gates * hidden, input_size),
                dtypes[1],
            )
            inputs_np[2] = from_float32(
                np.linspace(0.3, -0.2, num_dirs * gates * hidden * hidden, dtype=np.float32).reshape(num_dirs, gates * hidden, hidden),
                dtypes[2],
            )
            inputs_np[3] = from_float32(
                np.linspace(-0.2, 0.2, num_dirs * 2 * gates * hidden, dtype=np.float32).reshape(num_dirs, 2 * gates * hidden),
                dtypes[3],
            )
            seq_default = [seq_len] + [max(seq_len - 1 - (idx % 2), 1) for idx in range(1, batch)]
            inputs_np[4] = np.array(init_args.get("sequence_lens_value", seq_default), dtype=np.int64)
            init_values = np.linspace(0.1, -0.15, num_dirs * batch * hidden, dtype=np.float32).reshape(num_dirs, batch, hidden)
            inputs_np[5] = from_float32(init_values, dtypes[5])
            if op_name == "lstm":
                init_c_values = np.linspace(-0.05, 0.2, num_dirs * batch * hidden, dtype=np.float32).reshape(num_dirs, batch, hidden)
                peephole_values = np.linspace(-0.05, 0.05, num_dirs * 3 * hidden, dtype=np.float32).reshape(num_dirs, 3 * hidden)
                inputs_np[6] = from_float32(init_c_values, dtypes[6])
                inputs_np[7] = from_float32(peephole_values, dtypes[7])

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

        if op_name in {"random_uniform_like", "random_normal_like"}:
            # Like 型随机算子的输入只提供 shape 和默认 dtype，数值本身不会参与 reference 计算。
            pass

        if op_name == "bernoulli":
            # Bernoulli 使用固定概率样本，覆盖 0/1 极值和中间概率，保证随机输出可由 seed 复现。
            probs = np.asarray(
                init_args.get("prob_values", np.linspace(0.0, 1.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(probs, dtypes[0])

        if op_name == "multinomial":
            # Multinomial 使用固定概率矩阵，覆盖 one-hot、零概率和非归一化概率行。
            probs = np.asarray(
                init_args.get("prob_values", np.linspace(0.1, 1.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(probs, dtypes[0])

        if op_name == "binarizer":
            # Binarizer 使用固定样本覆盖阈值两侧和恰好等于阈值的严格大于边界。
            values = np.asarray(
                init_args.get("input_values", np.linspace(-1.0, 1.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])

        if op_name == "negative_log_likelihood_loss":
            # NLLLoss 使用固定 log-prob、标签和权重，覆盖 ignore_index、加权 mean、none/sum 和低精度写回。
            values = np.asarray(
                init_args.get("input_values", np.linspace(-0.25, -2.5, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = np.asarray(init_args.get("target_values", [0] * int(np.prod(shapes[1]))), dtype=np.int64).reshape(shapes[1])
            if len(inputs_np) > 2 and inputs_np[2] is not None:
                weights = np.asarray(init_args.get("weight_values", np.ones(shapes[2], dtype=np.float32)), dtype=np.float32).reshape(shapes[2])
                inputs_np[2] = from_float32(weights, dtypes[2])

        if op_name == "softmax_cross_entropy_loss":
            # SCE 使用固定 scores、标签和可选权重，覆盖 log_prob 多输出、ignore_index 和低精度写回。
            values = np.asarray(
                init_args.get("score_values", np.linspace(-1.5, 2.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = np.asarray(init_args.get("target_values", [0] * int(np.prod(shapes[1]))), dtype=np.int64).reshape(shapes[1])
            if len(inputs_np) > 2 and inputs_np[2] is not None:
                weights = np.asarray(init_args.get("weight_values", np.ones(shapes[2], dtype=np.float32)), dtype=np.float32).reshape(shapes[2])
                inputs_np[2] = from_float32(weights, dtypes[2])

        if op_name == "non_max_suppression":
            # NMS 使用固定 boxes/scores/thresholds，覆盖排序、score 阈值、IoU 抑制和 center_point_box。
            boxes = np.asarray(
                init_args.get("boxes_values", np.linspace(0.0, 1.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            scores = np.asarray(
                init_args.get("scores_values", np.linspace(0.9, 0.1, int(np.prod(shapes[1])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[1])
            inputs_np[0] = from_float32(boxes, dtypes[0])
            inputs_np[1] = from_float32(scores, dtypes[1])
            inputs_np[2] = np.asarray(init_args.get("max_output_value", 1), dtype=np.int64).reshape(shapes[2])
            inputs_np[3] = from_float32(np.asarray(init_args.get("iou_threshold_value", 0.0), dtype=np.float32).reshape(shapes[3]), dtypes[3])
            inputs_np[4] = from_float32(np.asarray(init_args.get("score_threshold_value", -np.inf), dtype=np.float32).reshape(shapes[4]), dtypes[4])

        if op_name == "dropout":
            # Dropout 使用固定样本、显式 ratio 和 training_mode，保证随机 mask 可由 seed 复现。
            values = np.asarray(
                init_args.get("input_values", np.linspace(0.0, 5.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            inputs_np[1] = np.asarray(init_args.get("ratio_value", 0.5), dtype=np.float32).reshape(shapes[1])
            inputs_np[2] = np.asarray(bool(init_args.get("training_mode_value", 1)), dtype=np.bool_).reshape(shapes[2])

        if op_name == "dynamic_quantize_linear":
            # DynamicQuantizeLinear 使用固定浮点样本，覆盖负数、零、正数和 min/max 包含 0 的官方缩放规则。
            values = np.asarray(
                init_args.get("input_values", np.linspace(-3.0, 6.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = values

        if op_name == "split":
            # Split 使用固定多列样本和非等分 split，覆盖多输出、axis 偏移和每个输出独立 shape。
            values = np.asarray(
                init_args.get("input_values", np.linspace(-3.0, 6.0, int(np.prod(shapes[0])), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(shapes[0])
            inputs_np[0] = from_float32(values, dtypes[0])
            if len(inputs_np) > 1:
                inputs_np[1] = np.asarray(init_args.get("split_value", [shapes[0][init_args.get("axis", 0)]]), dtype=np.int64)

        if op_name == "unique":
            # Unique 使用固定重复样本，分别覆盖首次出现顺序和按值排序后的 indices/inverse/counts。
            if dtypes[0] == "int64":
                values = np.asarray(init_args.get("input_values", [3, 1, 3, 2, 1, 3, -1, 2]), dtype=np.int64)
                inputs_np[0] = values.reshape(shapes[0])
            else:
                values = np.asarray(
                    init_args.get("input_values", [2.0, -1.0, 2.0, 0.5, -1.0, 3.0, 0.5, 4.0]),
                    dtype=np.float32,
                ).reshape(shapes[0])
                inputs_np[0] = from_float32(values, dtypes[0])

        inputs_tensor = []
        for data, d in zip(inputs_np, dtypes):
            if data is not None: inputs_tensor.append(Tensor(*data.shape, dtype=d, data=data))
            else: inputs_tensor.append(None)

        try:
            nps_result = run_nps_forward(op_cls, op_name, inputs_tensor, init_args, out_dtype)
            nps_out = nps_result.output
            nps_topk_indices = nps_result.topk_indices
        except Exception as e:
            print(f"  ❌ Iter {i} Crash: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        params_bin = build_cuda_params(op_name, inputs_np, init_args, shapes, dtypes, out_dtype, nps_out)

        if op_name in {"rnn", "gru", "lstm"}:
            recurrent_outputs = [np.asarray(out) for out in nps_out]
            y_np = recurrent_outputs[0]
            side_specs = [("Y_h", recurrent_outputs[1], f"tmp_{op_name}_y_h.bin")]
            if op_name == "lstm":
                side_specs.append(("Y_c", recurrent_outputs[2], "tmp_lstm_y_c.bin"))

            cuda_inputs = [
                np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float64)),
                np.ascontiguousarray(to_float32(inputs_np[1], dtypes[1]).astype(np.float64)),
                np.ascontiguousarray(to_float32(inputs_np[2], dtypes[2]).astype(np.float64)),
                np.ascontiguousarray(to_float32(inputs_np[3], dtypes[3]).astype(np.float64)),
                np.ascontiguousarray(inputs_np[4].astype(np.int64)),
                np.ascontiguousarray(to_float32(inputs_np[5], dtypes[5]).astype(np.float64)),
            ]
            if op_name == "lstm":
                cuda_inputs.extend(
                    [
                        np.ascontiguousarray(to_float32(inputs_np[6], dtypes[6]).astype(np.float64)),
                        np.ascontiguousarray(to_float32(inputs_np[7], dtypes[7]).astype(np.float64)),
                    ]
                )

            cuda_y = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=np.float64,
                target_shape=y_np.shape,
            )
            if cuda_y is None:
                continue

            missing_paths = [path for _name, _expected, path in side_specs if not os.path.exists(path)]
            if missing_paths:
                print(f"  ❌ Iter {i} FAILED")
                print(f"     Missing {op_cls.__name__} sidecar output: {', '.join(missing_paths)}")
                for _name, _expected, path in side_specs:
                    if os.path.exists(path):
                        os.remove(path)
                break

            comparisons = [("Y", y_np, cuda_y)]
            for name, expected, path in side_specs:
                cuda_side = np.fromfile(path, dtype=np.float64).reshape(expected.shape)
                os.remove(path)
                comparisons.append((name, expected, cuda_side))

            ok_all = True
            max_abs_all = 0.0
            max_rel_all = 0.0
            failed_name = None
            for name, expected, cuda_value in comparisons:
                cuda_ref = quantize_to_dtype_float32(cuda_value, out_dtype)
                expected_cmp = to_float32(expected, out_dtype)
                ok, cur_abs, cur_rel, _fail = check_accuracy(expected_cmp, cuda_ref, atol, rtol, out_dtype)
                max_abs_all = max(max_abs_all, cur_abs if cur_abs >= 0 else 0.0)
                max_rel_all = max(max_rel_all, cur_rel if cur_rel >= 0 else 0.0)
                if not ok and failed_name is None:
                    failed_name = name
                ok_all = ok_all and ok

            stats_abs.append(max_abs_all)
            stats_rel.append(max_rel_all)
            if ok_all:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                print(f"     {op_cls.__name__} {failed_name} mismatch")
                print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
                print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
                break
            continue

        if op_name == "dropout":
            y_np, mask_np = [np.asarray(out) for out in nps_out]
            cuda_inputs = [
                np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32)),
            ]
            cuda_y = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=np.float32,
                target_shape=y_np.shape,
            )
            if cuda_y is None:
                continue

            mask_path = "tmp_dropout_mask.bin"
            if not os.path.exists(mask_path):
                print(f"  ❌ Iter {i} FAILED")
                print("     Missing Dropout mask sidecar output")
                break
            cuda_mask = np.fromfile(mask_path, dtype=np.uint8).reshape(mask_np.shape).astype(np.bool_)
            os.remove(mask_path)

            nps_y = to_float32(y_np, out_dtype)
            cuda_y = quantize_to_dtype_float32(cuda_y, out_dtype)
            y_ok, max_abs, max_rel, _fail_mask = check_accuracy(nps_y, cuda_y, atol, rtol, out_dtype)
            mask_ok = np.array_equal(mask_np.astype(np.bool_), cuda_mask)

            stats_abs.append(max_abs if max_abs >= 0 else 0.0)
            stats_rel.append(max_rel if max_rel >= 0 else 0.0)
            if y_ok and mask_ok:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                if not y_ok:
                    print(f"     Dropout y mismatch: Max Abs Diff {max_abs:.6f}, Max Rel Diff {max_rel:.6f}")
                if not mask_ok:
                    print("     Dropout mask mismatch")
                break
            continue

        if op_name == "batch_normalization" and int(init_args.get("training_mode", 0)):
            y_np, running_mean_np, running_var_np = [np.asarray(out) for out in nps_out]
            cuda_inputs = [
                np.ascontiguousarray(to_float32(inputs_np[idx], dtypes[idx]).astype(np.float64))
                for idx in range(5)
            ]
            cuda_y = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=np.float64,
                target_shape=y_np.shape,
            )
            if cuda_y is None:
                continue

            side_paths = {
                "running_mean": "tmp_batch_norm_running_mean.bin",
                "running_var": "tmp_batch_norm_running_var.bin",
            }
            if not all(os.path.exists(path) for path in side_paths.values()):
                print(f"  ❌ Iter {i} FAILED")
                print("     Missing BatchNormalization training sidecar output")
                for path in side_paths.values():
                    if os.path.exists(path):
                        os.remove(path)
                break

            cuda_running_mean = np.fromfile(side_paths["running_mean"], dtype=np.float64).reshape(running_mean_np.shape)
            cuda_running_var = np.fromfile(side_paths["running_var"], dtype=np.float64).reshape(running_var_np.shape)
            for path in side_paths.values():
                os.remove(path)

            comparisons = [
                ("y", y_np, cuda_y),
                ("running_mean", running_mean_np, cuda_running_mean),
                ("running_var", running_var_np, cuda_running_var),
            ]
            ok_all = True
            max_abs_all = 0.0
            max_rel_all = 0.0
            failed_name = ""
            for name, expected, actual in comparisons:
                expected_f32 = to_float32(expected, out_dtype)
                actual_q = quantize_to_dtype_float32(actual, out_dtype)
                ok, max_abs, max_rel, _fail_mask = check_accuracy(expected_f32, actual_q, atol, rtol, out_dtype)
                max_abs_all = max(max_abs_all, max_abs if max_abs >= 0 else 0.0)
                max_rel_all = max(max_rel_all, max_rel if max_rel >= 0 else 0.0)
                if not ok:
                    ok_all = False
                    failed_name = name
                    break

            stats_abs.append(max_abs_all)
            stats_rel.append(max_rel_all)
            if ok_all:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                print(f"     BatchNormalization training {failed_name} mismatch")
                print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
                print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
                break
            continue

        if op_name == "layer_normalization" and int(init_args.get("emit_stats", 0)):
            y_np, mean_np, inv_std_np = [np.asarray(out) for out in nps_out]
            cuda_inputs = [
                np.ascontiguousarray(to_float32(inputs_np[idx], dtypes[idx]).astype(np.float64))
                for idx in range(3)
            ]
            cuda_y = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=np.float64,
                target_shape=y_np.shape,
            )
            if cuda_y is None:
                continue

            side_paths = {
                "mean": "tmp_layer_norm_mean.bin",
                "inv_std": "tmp_layer_norm_inv_std.bin",
            }
            if not all(os.path.exists(path) for path in side_paths.values()):
                print(f"  ❌ Iter {i} FAILED")
                print("     Missing LayerNormalization stats sidecar output")
                for path in side_paths.values():
                    if os.path.exists(path):
                        os.remove(path)
                break

            cuda_mean = np.fromfile(side_paths["mean"], dtype=np.float64).reshape(mean_np.shape)
            cuda_inv_std = np.fromfile(side_paths["inv_std"], dtype=np.float64).reshape(inv_std_np.shape)
            for path in side_paths.values():
                os.remove(path)

            stash_dtype = nn.onnx_dtype_mapping.get(int(init_args.get("stash_type", 1)), "float32")
            comparisons = [
                ("y", y_np, cuda_y, out_dtype),
                ("mean", mean_np, cuda_mean, stash_dtype),
                ("inv_std", inv_std_np, cuda_inv_std, stash_dtype),
            ]
            ok_all = True
            max_abs_all = 0.0
            max_rel_all = 0.0
            failed_name = ""
            for name, expected, actual, dtype_name in comparisons:
                expected_f32 = to_float32(expected, dtype_name)
                actual_q = quantize_to_dtype_float32(actual, dtype_name)
                ok, max_abs, max_rel, _fail_mask = check_accuracy(expected_f32, actual_q, atol, rtol, dtype_name)
                max_abs_all = max(max_abs_all, max_abs if max_abs >= 0 else 0.0)
                max_rel_all = max(max_rel_all, max_rel if max_rel >= 0 else 0.0)
                if not ok:
                    ok_all = False
                    failed_name = name
                    break

            stats_abs.append(max_abs_all)
            stats_rel.append(max_rel_all)
            if ok_all:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                print(f"     LayerNormalization {failed_name} mismatch")
                print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
                print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
                break
            continue

        if op_name == "softmax_cross_entropy_loss":
            if isinstance(nps_out, list):
                loss_np = np.asarray(nps_out[0])
                log_prob_np = np.asarray(nps_out[1])
            else:
                loss_np = np.asarray(nps_out)
                log_prob_np = None
            loss_shape = loss_np.shape if loss_np.shape != () else (1,)
            loss_cmp = loss_np.reshape(loss_shape)
            cuda_inputs = [
                np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float64)),
                np.ascontiguousarray(inputs_np[1].astype(np.int64)),
                None if len(inputs_np) <= 2 or inputs_np[2] is None else np.ascontiguousarray(to_float32(inputs_np[2], dtypes[2]).astype(np.float64)),
            ]
            cuda_loss = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=np.float64,
                target_shape=loss_shape,
            )
            if cuda_loss is None:
                continue

            loss_ref = quantize_to_dtype_float32(cuda_loss, out_dtype)
            loss_nps = to_float32(loss_cmp, out_dtype)
            loss_ok, loss_abs, loss_rel, _loss_fail = check_accuracy(loss_nps, loss_ref, atol, rtol, out_dtype)

            log_ok = True
            log_abs = 0.0
            log_rel = 0.0
            if log_prob_np is not None:
                log_path = "tmp_out_log_prob.bin"
                if not os.path.exists(log_path):
                    print(f"  ❌ Iter {i} FAILED")
                    print("     Missing SoftmaxCrossEntropyLoss log_prob sidecar output")
                    break
                cuda_log = np.fromfile(log_path, dtype=np.float64).reshape(log_prob_np.shape)
                os.remove(log_path)
                log_ref = quantize_to_dtype_float32(cuda_log, out_dtype)
                log_nps = to_float32(log_prob_np, out_dtype)
                log_ok, log_abs, log_rel, _log_fail = check_accuracy(log_nps, log_ref, atol, rtol, out_dtype)

            max_abs = max(loss_abs if loss_abs >= 0 else 0.0, log_abs if log_abs >= 0 else 0.0)
            max_rel = max(loss_rel if loss_rel >= 0 else 0.0, log_rel if log_rel >= 0 else 0.0)
            stats_abs.append(max_abs)
            stats_rel.append(max_rel)
            if loss_ok and log_ok:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                if not loss_ok:
                    print(f"     SCE loss mismatch: Max Abs Diff {loss_abs:.6f}, Max Rel Diff {loss_rel:.6f}")
                if not log_ok:
                    print(f"     SCE log_prob mismatch: Max Abs Diff {log_abs:.6f}, Max Rel Diff {log_rel:.6f}")
                break
            continue

        if op_name == "dynamic_quantize_linear":
            y_np, scale_np, zp_np = nps_out
            y_np = np.asarray(y_np, dtype=np.uint8)
            scale_np = np.asarray(scale_np, dtype=np.float32).reshape(())
            zp_np = np.asarray(zp_np, dtype=np.uint8).reshape(())

            flat_len = int(y_np.size)
            cuda_inputs = [np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32))]
            cuda_out = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=np.float32,
                target_shape=(flat_len + 2,),
            )
            if cuda_out is None:
                continue

            cuda_flat = np.asarray(cuda_out, dtype=np.float32).reshape(-1)
            cuda_y = np.rint(cuda_flat[:flat_len]).clip(0, 255).astype(np.uint8).reshape(y_np.shape)
            cuda_scale = np.asarray(cuda_flat[flat_len], dtype=np.float32).reshape(())
            cuda_zp = np.asarray(np.rint(cuda_flat[flat_len + 1]).clip(0, 255), dtype=np.uint8).reshape(())

            y_ok = np.array_equal(y_np, cuda_y)
            scale_abs = float(abs(float(scale_np) - float(cuda_scale)))
            scale_rel = scale_abs / max(abs(float(cuda_scale)), 1e-12)
            scale_ok = scale_abs <= 1e-7 + 1e-6 * abs(float(cuda_scale))
            zp_ok = int(zp_np) == int(cuda_zp)

            y_abs = float(np.max(np.abs(y_np.astype(np.int16) - cuda_y.astype(np.int16)))) if y_np.size else 0.0
            zp_abs = float(abs(int(zp_np) - int(cuda_zp)))
            max_abs = max(y_abs, scale_abs, zp_abs)
            max_rel = scale_rel
            stats_abs.append(max_abs)
            stats_rel.append(max_rel)

            if y_ok and scale_ok and zp_ok:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                if not y_ok:
                    print(f"     y mismatch, max uint8 diff: {y_abs:.0f}")
                if not scale_ok:
                    print(f"     y_scale mismatch: CUDA={float(cuda_scale):.9g}, C={float(scale_np):.9g}")
                if not zp_ok:
                    print(f"     y_zero_point mismatch: CUDA={int(cuda_zp)}, C={int(zp_np)}")
                break
            continue

        if op_name == "split":
            flat_outputs = [np.asarray(out) for out in nps_out]
            flat_len = int(sum(out.size for out in flat_outputs))
            cuda_inputs = [
                np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32)),
            ]
            if len(inputs_np) > 1:
                cuda_inputs.append(np.ascontiguousarray(inputs_np[1].astype(np.int64)))
            cuda_out = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=np.float32,
                target_shape=(flat_len,),
            )
            if cuda_out is None:
                continue

            cuda_flat = np.asarray(cuda_out, dtype=np.float32).reshape(-1)
            offset = 0
            ok_all = True
            max_abs_all = 0.0
            max_rel_all = 0.0
            failed_index = -1
            for out_idx, expected_piece in enumerate(flat_outputs):
                piece_len = int(expected_piece.size)
                cuda_piece = cuda_flat[offset:offset + piece_len].reshape(expected_piece.shape)
                offset += piece_len
                nps_piece = to_float32(expected_piece, out_dtype)
                cuda_piece = quantize_to_dtype_float32(cuda_piece, out_dtype)
                ok_piece, max_abs, max_rel, _fail_mask = check_accuracy(nps_piece, cuda_piece, atol, rtol, out_dtype)
                max_abs_all = max(max_abs_all, max_abs if max_abs >= 0 else 0.0)
                max_rel_all = max(max_rel_all, max_rel if max_rel >= 0 else 0.0)
                if not ok_piece:
                    ok_all = False
                    failed_index = out_idx
                    break

            stats_abs.append(max_abs_all)
            stats_rel.append(max_rel_all)
            if ok_all:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                print(f"     Split output {failed_index} mismatch")
                print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
                print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
                break
            continue

        if op_name == "unique":
            values_np, indices_np, inverse_np, counts_np = [np.asarray(out) for out in nps_out]
            input_arr = inputs_np[0]
            if dtypes[0] == "int64":
                cuda_inputs = [np.ascontiguousarray(input_arr.astype(np.int64))]
                cuda_value_dtype = np.int64
            else:
                cuda_inputs = [np.ascontiguousarray(to_float32(input_arr, dtypes[0]).astype(np.float32))]
                cuda_value_dtype = np.float32

            cuda_values = run_cuda_ground_truth(
                op_name,
                cuda_inputs,
                params_binary=params_bin,
                output_dtype=cuda_value_dtype,
                target_shape=values_np.shape,
            )
            if cuda_values is None:
                continue

            side_paths = {
                "indices": "tmp_unique_indices.bin",
                "inverse": "tmp_unique_inverse.bin",
                "counts": "tmp_unique_counts.bin",
            }
            if not all(os.path.exists(path) for path in side_paths.values()):
                print(f"  ❌ Iter {i} FAILED")
                print("     Missing Unique sidecar output")
                for path in side_paths.values():
                    if os.path.exists(path):
                        os.remove(path)
                break

            cuda_indices = np.fromfile(side_paths["indices"], dtype=np.int64).reshape(indices_np.shape)
            cuda_inverse = np.fromfile(side_paths["inverse"], dtype=np.int64).reshape(inverse_np.shape)
            cuda_counts = np.fromfile(side_paths["counts"], dtype=np.int64).reshape(counts_np.shape)
            for path in side_paths.values():
                os.remove(path)

            if out_dtype == "int64":
                values_ok = np.array_equal(values_np.astype(np.int64), cuda_values.astype(np.int64))
                value_abs = 0.0 if values_ok else -1.0
                value_rel = 0.0 if values_ok else -1.0
            else:
                nps_values = to_float32(values_np, out_dtype)
                cuda_values = quantize_to_dtype_float32(cuda_values, out_dtype)
                values_ok, value_abs, value_rel, _fail_mask = check_accuracy(nps_values, cuda_values, atol, rtol, out_dtype)

            indices_ok = np.array_equal(indices_np.astype(np.int64), cuda_indices)
            inverse_ok = np.array_equal(inverse_np.astype(np.int64), cuda_inverse)
            counts_ok = np.array_equal(counts_np.astype(np.int64), cuda_counts)
            max_abs = value_abs if value_abs >= 0 else 0.0
            max_rel = value_rel if value_rel >= 0 else 0.0
            stats_abs.append(max_abs)
            stats_rel.append(max_rel)

            if values_ok and indices_ok and inverse_ok and counts_ok:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                if not values_ok:
                    print("     Unique values mismatch")
                if not indices_ok:
                    print("     Unique indices mismatch")
                if not inverse_ok:
                    print("     Unique inverse mismatch")
                if not counts_ok:
                    print("     Unique counts mismatch")
                break
            continue

        # 4. 数据转换与 广播处理
        expected_shape = nps_out.shape
        if expected_shape == ():
            expected_shape = (1,) # 统一当成 1 元素张量来跑 CUDA/读写 bin
            nps_out = np.array([nps_out], dtype=nps_out.dtype)
        is_complex_kernel = op_name in ["conv2d", "conv_integer", "qlinear_conv", "conv_transpose", "col2im", "deform_conv", "attention", "matmul_integer", "qlinear_matmul", "max_pool", "average_pool", "lp_pool", "global_average_pool", "global_max_pool", "global_lp_pool", "lrn", "mean_variance_normalization", "batch_normalization", "instance_normalization", "layer_normalization", "lp_normalization", "group_normalization", "negative_log_likelihood_loss", "softmax_cross_entropy_loss", "non_max_suppression", "max_unpool", "grid_sample", "max_roi_pool", "roi_align", "dft", "stft", "rnn", "gru", "lstm", "gemm", "softmax", "hardmax", "log_softmax"] # 这些算子自己处理形状
        is_double_kernel = is_complex_kernel or op_name in ["quantize_linear", "dequantize_linear"]
        int64_passthrough_ops = {
            "gather", "scatternd", "tensor_scatter", "scatter_elements", "gather_elements", "gathernd",
            "resize", "affine_grid", "topk", "max_unpool", "roi_align", "col2im", "dft", "stft",
            "rnn", "gru", "lstm", "tile", "expand", "pad", "center_crop_pad", "slice",
            "constant_of_shape", "rotary_embedding", "tril", "triu", "trilu",
            "hann_window", "hamming_window", "blackman_window",
            "range", "one_hot", "reverse_sequence", "mel_weight_matrix",
            "negative_log_likelihood_loss", "softmax_cross_entropy_loss", "non_max_suppression",
        }
        no_broadcast_ops = {
            "matmul", "reduce_mean", "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
            "reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square",
            "gather", "gather_elements", "gathernd", "scatternd", "tensor_scatter", "scatter_elements",
            "nonzero", "argmin", "argmax", "size", "resize", "affine_grid", "grid_sample", "einsum",
            "topk", "random_uniform", "random_uniform_like", "random_normal", "random_normal_like", "bernoulli", "multinomial",
            "expand", "flatten", "reshape", "squeeze", "unsqueeze",
            "transpose", "tile", "concat", "pad", "center_crop_pad", "depth_to_space", "space_to_depth",
            "slice", "compress", "constant_of_shape", "eye_like", "rotary_embedding", "col2im",
            "deform_conv", "attention", "tril", "triu", "trilu",
            "hann_window", "hamming_window", "blackman_window",
            "range", "one_hot", "reverse_sequence", "det", "mel_weight_matrix",
            "negative_log_likelihood_loss", "softmax_cross_entropy_loss", "non_max_suppression",
            "quantize_linear", "dequantize_linear",
        }
        
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

                if op_name in int64_passthrough_ops and d == "int64":
                    cuda_inputs.append(np.ascontiguousarray(inp.astype(np.int64)))
                    continue

                target_dtype = np.float64 if is_double_kernel else np.float32
                if op_name in {"quantize_linear", "dequantize_linear"} and init_args.get("omit_zero_point") and i == 2 and d == "float8_e8m0":
                    val_f32 = np.zeros(np.asarray(inp).shape, dtype=np.float32)
                else:
                    val_f32 = to_float32(inp, d)
                
                # 广播逻辑
                if (not is_complex_kernel) and (op_name not in no_broadcast_ops):
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
                            if (not is_complex_kernel) and (op_name not in ["matmul", "reduce_mean", "gather", "scatternd", "tensor_scatter", "scatter_elements","nonzero", "argmin", "argmax", "size", "resize", "affine_grid", "grid_sample", "einsum", "topk", "random_uniform", "random_uniform_like", "random_normal", "random_normal_like", "bernoulli", "multinomial", "expand", "flatten", "reshape", "squeeze", "unsqueeze", "transpose", "tile", "concat", "pad", "center_crop_pad", "depth_to_space", "space_to_depth", "slice", "compress", "constant_of_shape", "eye_like", "rotary_embedding", "col2im", "deform_conv", "attention", "negative_log_likelihood_loss", "softmax_cross_entropy_loss", "non_max_suppression"]):
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
