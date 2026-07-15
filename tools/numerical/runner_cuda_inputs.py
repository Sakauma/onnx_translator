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
    cuda_inputs = []
    for input_index, (input_value, dtype_name) in enumerate(zip(inputs_np, dtypes)):
        if input_value is None:
            cuda_inputs.append(None)
            continue
        if op_name == "bitcast":
            cuda_inputs.append(
                np.ascontiguousarray(input_value.astype(nn.DTYPE_TO_NUMPY[dtype_name], copy=False))
            )
            continue
        if op_name in BITWISE_OPS and dtype_name == "int32":
            cuda_inputs.append(np.ascontiguousarray(input_value.astype(np.int32, copy=False)))
            continue
        if config.int64_passthrough and dtype_name == "int64":
            cuda_inputs.append(np.ascontiguousarray(input_value.astype(np.int64)))
            continue

        if (
            op_name in QUANTIZATION_OPS
            and init_args.get("omit_zero_point")
            and input_index == 2
            and dtype_name == "float8_e8m0"
        ):
            converted = np.zeros(np.asarray(input_value).shape, dtype=np.float32)
        else:
            converted = to_float32(input_value, dtype_name)

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
