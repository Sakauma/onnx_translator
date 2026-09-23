"""Checker-valid public graph regressions for integer Einsum and float8 quantization."""

import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

import nn
from nn import Graph, Tensor
from nn.ONNXImport import ONNXImport


def _tensor(values, dtype):
    values = np.asarray(values)
    return Tensor(*values.shape, dtype=dtype, data=values)


def _run_checked_graph(tmp_path, op_type, inputs, input_protos, output_proto, output_shape, attrs, opset):
    names = list(inputs)
    node = helper.make_node(op_type, names, ["y"], **attrs)
    model = helper.make_model(
        helper.make_graph(
            [node],
            f"native_precision_{op_type}",
            [helper.make_tensor_value_info(name, proto, list(np.asarray(inputs[name]).shape))
             for name, proto in zip(names, input_protos)],
            [helper.make_tensor_value_info("y", output_proto, list(output_shape))],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=9,
    )
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / f"{op_type}.onnx"
    onnx.save(model, path)
    expected = ReferenceEvaluator(model).run(None, inputs)[0]
    imported = ONNXImport(str(path), strict=True)
    actual = Graph(imported, names, ["y"]).forward(
        *(_tensor(inputs[name], nn.onnx_dtype_mapping[proto]) for name, proto in zip(names, input_protos))
    )
    return actual.data, expected


@pytest.mark.parametrize("dtype,proto", [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)])
@pytest.mark.parametrize("equation,operand_values", [
    ("i->i", [[2**53 + 1, 2**60 + 3]]),
    ("ij->ji", [[[2**53 + 1, 2**53 + 3], [2**60 + 1, 2**60 + 3]]]),
    ("i,i->i", [[2**53 + 1, 2**60 + 3], [3, 2]]),
    ("i->", [[2**53 + 1, 2**53 + 3, 5]]),
    ("ij,jk->ik", [[[2**53 + 1, 3], [5, 2**53 + 3]], [[1, 2], [3, 1]]]),
])
def test_integer_einsum_graph_matches_exact_onnx_reference(
    tmp_path, dtype, proto, equation, operand_values
):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    inputs = {f"x{i}": np.asarray(values, dtype=np.dtype(dtype)) for i, values in enumerate(operand_values)}
    expected_shape = np.einsum(equation, *inputs.values()).shape
    actual, expected = _run_checked_graph(
        tmp_path, "Einsum", inputs, [proto] * len(inputs), proto, expected_shape,
        {"equation": equation}, 17,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype,proto", [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)])
def test_integer_einsum_reduction_wrap_matches_onnx_reference(tmp_path, dtype, proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    maximum = np.iinfo(np.dtype(dtype)).max
    x = np.asarray([maximum, 1], dtype=np.dtype(dtype))
    actual, expected = _run_checked_graph(
        tmp_path, "Einsum", {"x": x}, [proto], proto, (), {"equation": "i->"}, 17,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype,proto,unit", [
    ("float8_e4m3", TensorProto.FLOAT8E4M3FN, 2.0**-9),
    ("float8_e5m2", TensorProto.FLOAT8E5M2, 2.0**-16),
])
@pytest.mark.parametrize("saturate", [0, 1])
def test_quantize_float8_subnormal_rounding_matches_reference(
    tmp_path, dtype, proto, unit, saturate
):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    half = np.float32(unit / 2)
    below_half = np.nextafter(half, np.float32(0), dtype=np.float32)
    above_half = np.nextafter(half, np.float32(np.inf), dtype=np.float32)
    maximum_subnormal = 7 if dtype == "float8_e4m3" else 3
    x = np.asarray([
        0.0, -0.0, below_half, half, above_half, unit, 1.5 * unit,
        maximum_subnormal * unit, (maximum_subnormal + 0.5) * unit,
        (maximum_subnormal + 1) * unit,
        -below_half, -half, -above_half, -unit,
        -maximum_subnormal * unit, -(maximum_subnormal + 0.5) * unit,
    ], dtype=np.float32)
    scale = np.asarray(1.0, dtype=np.float32)
    actual, expected = _run_checked_graph(
        tmp_path, "QuantizeLinear", {"x": x, "scale": scale},
        [TensorProto.FLOAT, TensorProto.FLOAT], proto, x.shape,
        {"output_dtype": proto, "saturate": saturate}, 25,
    )
    np.testing.assert_array_equal(actual, expected.view(np.uint8))
