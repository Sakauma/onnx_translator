# /**
#   ******************************************************************************
#   * @file        runner_inputs.py
#   * @author      Egor Izmaylov
#   * @brief       为数值验证计划生成通用输入，并应用按算子划分的边界样本策略。
#   * @details     2026.07.15  V1.0.0  从 runner.py 拆分输入样本准备职责
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np

import nn

from .data import generate_random_data
from .dtype import from_float32, to_float32


def prepare_input_samples(op_name, shapes, dtypes, init_args):
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

    return inputs_np
