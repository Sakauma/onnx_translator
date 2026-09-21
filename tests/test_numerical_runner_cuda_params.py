# /**
#   ******************************************************************************
#   * @file        test_numerical_runner_cuda_params.py
#   * @author      Egor Izmaylov
#   * @brief       Covers CUDA verifier parameter payload construction.
#   * @details     2026.06.28  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np
import pytest

from tools.numerical.runner_cuda_params import build_cuda_params


def _i32(payload):
    return np.frombuffer(payload, dtype=np.int32).tolist()


def test_matmul_params_encode_mkn():
    payload = build_cuda_params(
        "matmul",
        [np.zeros((2, 3), dtype=np.float32), np.zeros((3, 4), dtype=np.float32)],
        {},
        [(2, 3), (3, 4)],
        ["float32", "float32"],
        "float32",
        np.zeros((2, 4), dtype=np.float32),
    )

    assert _i32(payload) == [2, 3, 4]


def test_softmax_params_resolve_negative_axis():
    payload = build_cuda_params(
        "softmax",
        [np.zeros((2, 3, 4), dtype=np.float32)],
        {"axis": -1},
        [(2, 3, 4)],
        ["float32"],
        "float32",
        np.zeros((2, 3, 4), dtype=np.float32),
    )

    assert _i32(payload) == [6, 4, 1]


def test_quantize_linear_params_encode_dtype_axis_block_and_shapes():
    payload = build_cuda_params(
        "quantize_linear",
        [
            np.zeros((2, 3), dtype=np.float32),
            np.ones((3,), dtype=np.float32),
            np.zeros((3,), dtype=np.int8),
        ],
        {"axis": -1, "block_size": 2, "saturate": 0},
        [(2, 3), (3,), (3,)],
        ["float32", "float32", "int8"],
        "int4",
        np.zeros((2, 3), dtype=np.int8),
    )

    assert _i32(payload) == [9, 1, 0, 2, 1, 3, 3, 2, 1, 2, 3]


def test_slice_params_encode_output_shape_starts_and_steps():
    payload = build_cuda_params(
        "slice",
        [
            np.zeros((2, 3, 4), dtype=np.float32),
            np.array([-1], dtype=np.int64),
            np.array([-5], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.array([-1], dtype=np.int64),
        ],
        {"starts_value": [-1], "ends_value": [-5], "axes_value": [-1], "steps_value": [-1]},
        [(2, 3, 4), (1,), (1,), (1,), (1,)],
        ["float32", "int64", "int64", "int64", "int64"],
        "float32",
        np.zeros((2, 3, 4), dtype=np.float32),
    )

    assert _i32(payload) == [3, 2, 3, 4, 2, 3, 4, 0, 0, 3, 1, 1, -1]


@pytest.mark.parametrize(("reduction", "code"), [("max", 3), ("min", 4)])
def test_scatter_elements_params_encode_opset18_reductions(reduction, code):
    inputs = [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.int64),
        np.ones((2, 3), dtype=np.float32),
    ]
    payload = build_cuda_params(
        "scatter_elements",
        inputs,
        {"axis": 1, "reduction": reduction, "version": "18"},
        [(2, 3), (2, 3), (2, 3)],
        ["float32", "int64", "float32"],
        "float32",
    )
    assert _i32(payload)[:3] == [2, 1, code]


def test_scatter_elements_params_reject_unknown_reduction():
    inputs = [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.int64),
        np.ones((2, 3), dtype=np.float32),
    ]
    with pytest.raises(ValueError, match="unsupported ScatterElements reduction"):
        build_cuda_params(
            "scatter_elements",
            inputs,
            {"axis": 1, "reduction": "mystery"},
            [(2, 3), (2, 3), (2, 3)],
            ["float32", "int64", "float32"],
            "float32",
        )
