# /**
#   ******************************************************************************
#   * @file        test_numerical_runner_architecture.py
#   * @author      Egor Izmaylov
#   * @brief       验证数值调度器拆分后的算子族配置和 CUDA 输入策略。
#   * @details     2026.07.15  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np
import pytest

from tools.numerical.runner_config import resolve_verification_config
from tools.numerical.runner_cuda_inputs import build_cuda_inputs, resolve_cuda_output_dtype


@pytest.mark.parametrize(
    "op_name,out_dtype,expected",
    [
        ("add", "float32", (1e-4, 1e-4)),
        ("cos", "float32", (0.02, 1e-4)),
        ("einsum", "float32", (1e-2, 1e-3)),
        ("add", "float16", (0.01, 0.01)),
        ("add", "bfloat16", (0.1, 0.02)),
        ("add", "float8_e4m3", (0.1, 0.1)),
        ("gather", "int64", (0.0, 0.0)),
    ],
)
def test_verification_tolerances_are_resolved_from_config(op_name, out_dtype, expected):
    config = resolve_verification_config(op_name, out_dtype)

    assert (config.atol, config.rtol) == expected


def test_operator_family_config_controls_cuda_conversion():
    conv_config = resolve_verification_config("conv2d", "float32")
    gather_config = resolve_verification_config("gather", "int64")
    add_config = resolve_verification_config("add", "float32")

    assert conv_config.complex_kernel
    assert conv_config.double_kernel
    assert gather_config.int64_passthrough
    assert not gather_config.broadcast_inputs
    assert add_config.broadcast_inputs

    broadcast = build_cuda_inputs(
        "add",
        [np.asarray([1.0, 2.0], dtype=np.float32)],
        ["float32"],
        {},
        (2, 2),
        add_config,
    )[0]
    passthrough = build_cuda_inputs(
        "gather",
        [np.asarray([1, 2], dtype=np.int64)],
        ["int64"],
        {},
        (2,),
        gather_config,
    )[0]

    assert broadcast.shape == (2, 2)
    assert broadcast.dtype == np.float32
    assert passthrough.dtype == np.int64
    assert resolve_cuda_output_dtype("gather", "int64", gather_config) == np.int64
