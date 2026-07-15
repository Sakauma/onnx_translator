# /**
#   ******************************************************************************
#   * @file        runner_cuda_inputs.py
#   * @author      Egor Izmaylov
#   * @brief       按验证配置转换 CUDA 输入缓冲区并选择参考程序输出类型。
#   * @details     2026.07.15  V1.0.0  从 runner.py 拆分 CUDA 数据转换职责
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np

import nn

from .dtype import to_float32


BITWISE_OPS = frozenset({"bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "bit_shift"})
QUANTIZATION_OPS = frozenset({"quantize_linear", "dequantize_linear"})


def build_cuda_inputs(op_name, inputs_np, dtypes, init_args, expected_shape, config):
    """把本轮 NumPy 输入转换为 CUDA verifier 约定的连续缓冲区。

    普通算子以 float32 交换数据，复杂 kernel 和量化算子按配置使用 float64。
    位模式、位运算和 int64 索引必须绕过浮点转换，否则大整数或 NaN payload
    会在进入 verifier 前发生不可逆变化。
    """
    cuda_inputs = []
    for input_index, (input_value, dtype_name) in enumerate(zip(inputs_np, dtypes)):
        if input_value is None:
            cuda_inputs.append(None)
            continue
        # BitCast 验证的是存储位模式，不能经过 to_float32 的数值解释。
        if op_name == "bitcast":
            cuda_inputs.append(
                np.ascontiguousarray(input_value.astype(nn.DTYPE_TO_NUMPY[dtype_name], copy=False))
            )
            continue
        if op_name in BITWISE_OPS and dtype_name == "int32":
            cuda_inputs.append(np.ascontiguousarray(input_value.astype(np.int32, copy=False)))
            continue
        # 索引、shape 和轴参数保持 int64，以免 float32 丢失大整数精度。
        if config.int64_passthrough and dtype_name == "int64":
            cuda_inputs.append(np.ascontiguousarray(input_value.astype(np.int64)))
            continue

        # 可选 zero_point 在 C 路径中省略，但 CUDA 参数槽仍需等形零缓冲区占位。
        if (
            op_name in QUANTIZATION_OPS
            and init_args.get("omit_zero_point")
            and input_index == 2
            and dtype_name == "float8_e8m0"
        ):
            converted = np.zeros(np.asarray(input_value).shape, dtype=np.float32)
        else:
            converted = to_float32(input_value, dtype_name)

        # 普通逐元素 verifier 按输出 shape 线性读取输入，因此在写 bin 前完成广播。
        if not config.complex_kernel and config.broadcast_inputs:
            try:
                if converted.shape != expected_shape:
                    converted = np.broadcast_to(converted, expected_shape)
            except Exception as exc:
                print(f"Warning: Broadcast failed for input {input_index} in {op_name}: {exc}")

        target_dtype = np.float64 if config.double_kernel else np.float32
        cuda_inputs.append(converted.astype(target_dtype))
    return cuda_inputs


def resolve_cuda_output_dtype(op_name, out_dtype, config):
    """选择读取 CUDA 主输出文件时使用的 NumPy dtype。"""

    if out_dtype == "bool":
        return np.uint8
    if op_name in {"qlinear_conv", "qlinear_matmul"} and out_dtype == "uint8":
        return np.uint8
    if out_dtype == "int32":
        return np.int32
    if out_dtype == "int64":
        return np.int64
    if op_name == "bitcast":
        return nn.DTYPE_TO_NUMPY[out_dtype]
    return np.float64 if config.double_kernel else np.float32
