# /**
#   ******************************************************************************
#   * @file        test_quantize_saturation_regression.py
#   * @author      Egor Izmaylov
#   * @brief       回归验证 QuantizeLinear 在中间溢出和 nearest-even 边界下的整数饱和语义。
#   * @details     2026.09.21  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

import nn
from nn import Tensor
from nn.Operators import QuantizeLinear


def _tensor(values, dtype):
    data = np.asarray(values, dtype=nn.DTYPE_TO_NUMPY[dtype])
    return Tensor(*data.shape, dtype=dtype, data=data)


def _quantize(x, scale, zero_point, input_dtype, output_dtype, opset):
    return QuantizeLinear(
        ["x", "scale", "zero_point"],
        ["y"],
        dtype=output_dtype,
        version=str(opset),
    ).forward(
        _tensor(x, input_dtype),
        _tensor(scale, input_dtype),
        _tensor(zero_point, output_dtype),
    )["tensor"].data


def _reference_quantize(x, scale, zero_point, output_proto, opset):
    graph = helper.make_graph(
        [helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"])],
        "quantize_saturation_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, list(scale.shape)),
            helper.make_tensor_value_info("zero_point", output_proto, list(zero_point.shape)),
        ],
        [helper.make_tensor_value_info("y", output_proto, list(x.shape))],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return ReferenceEvaluator(model).run(
        None,
        {"x": x, "scale": scale, "zero_point": zero_point},
    )[0]


@pytest.fixture(autouse=True)
def _require_c_backend():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")


@pytest.mark.parametrize(
    "output_dtype,zero_point,expected",
    [
        ("int8", np.array([0], dtype=np.int8), np.array([127, -128, 127, -128], dtype=np.int8)),
        ("uint8", np.array([0], dtype=np.uint8), np.array([255, 0, 255, 0], dtype=np.uint8)),
    ],
)
def test_float32_division_overflow_and_finite_out_of_range_saturate_at_opset17(
    output_dtype, zero_point, expected
):
    x = np.array(
        [np.finfo(np.float32).max, -np.finfo(np.float32).max, 200.0, -200.0],
        dtype=np.float32,
    )
    scale = np.array([np.float32(1.0e-20)], dtype=np.float32)

    actual = _quantize(x, scale, zero_point, "float32", output_dtype, opset=17)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "output_dtype,zero_point,expected",
    [
        ("int8", np.array([0], dtype=np.int8), np.array([127, -128], dtype=np.int8)),
        ("uint8", np.array([0], dtype=np.uint8), np.array([255, 0], dtype=np.uint8)),
    ],
)
def test_float16_intermediate_overflow_saturates_at_opset19(output_dtype, zero_point, expected):
    x = np.array([696.0, -696.0], dtype=np.float16)
    scale = np.array([np.float16(0.01000213623046875)], dtype=np.float16)

    actual = _quantize(x, scale, zero_point, "float16", output_dtype, opset=19)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "output_dtype,output_proto,zero_point",
    [
        ("int8", TensorProto.INT8, np.array([3], dtype=np.int8)),
        ("uint8", TensorProto.UINT8, np.array([3], dtype=np.uint8)),
    ],
)
def test_finite_nearest_even_rounding_precedes_odd_zero_point_and_saturation(
    output_dtype, output_proto, zero_point
):
    quotients = np.array(
        [-129.0, -128.5, -127.5, -2.5, -1.5, 1.5, 2.5, 126.5, 127.5, 256.0],
        dtype=np.float32,
    )
    scale = np.array([1.0], dtype=np.float32)
    expected = _reference_quantize(quotients, scale, zero_point, output_proto, opset=17)

    actual = _quantize(quotients, scale, zero_point, "float32", output_dtype, opset=17)

    np.testing.assert_array_equal(actual, expected)
    odd_tie_index = int(np.flatnonzero(quotients == np.float32(2.5))[0])
    assert int(actual[odd_tie_index]) == 5
