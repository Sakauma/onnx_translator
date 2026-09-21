# /**
#   ******************************************************************************
#   * @file        test_followup_native_precision.py
#   * @author      Egor Izmaylov
#   * @brief       回归验证纯数据移动精度与 Scatter opset reduction 边界。
#   * @details     2026.09.21  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import nn
from nn import Graph, Tensor
from nn.Operators import (
    DepthToSpace,
    Resize,
    ReverseSequence,
    ScatterElements,
    ScatterND,
    SpaceToDepth,
    Transpose,
    Trilu,
)
from nn.importer import ONNXImport


def _tensor(data, dtype):
    data = np.ascontiguousarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


def _large_values(dtype):
    if dtype == "int64":
        return np.array(
            [2**53 + 1, 2**60 + 3, np.iinfo(np.int64).max, np.iinfo(np.int64).min],
            dtype=np.int64,
        )
    return np.array(
        [2**53 + 1, 2**63 + 1, np.iinfo(np.uint64).max, 0],
        dtype=np.uint64,
    )


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_native_layout_movement_preserves_64_bit_integer_payloads(dtype):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = _large_values(dtype)
    matrix = values.reshape(2, 2)

    transposed = Transpose(["x"], ["y"], perm=[1, 0], dtype=dtype).forward(
        _tensor(matrix, dtype)
    )["tensor"].data
    np.testing.assert_array_equal(transposed, matrix.T)

    triangular = Trilu(["x"], ["y"], upper=1, dtype=dtype).forward(
        _tensor(matrix, dtype)
    )["tensor"].data
    np.testing.assert_array_equal(triangular, np.triu(matrix))

    sequence = values.reshape(2, 2, 1)
    sequence_lens = np.array([2, 2], dtype=np.int64)
    reversed_sequence = ReverseSequence(
        ["x", "sequence_lens"], ["y"], time_axis=0, batch_axis=1, dtype=dtype
    ).forward(_tensor(sequence, dtype), _tensor(sequence_lens, "int64"))["tensor"].data
    np.testing.assert_array_equal(reversed_sequence, sequence[::-1])

    spatial = values.reshape(1, 1, 2, 2)
    space_to_depth = SpaceToDepth(["x"], ["y"], blocksize=2, dtype=dtype).forward(
        _tensor(spatial, dtype)
    )["tensor"].data
    np.testing.assert_array_equal(space_to_depth, values.reshape(1, 4, 1, 1))

    depth = values.reshape(1, 4, 1, 1)
    depth_to_space = DepthToSpace(
        ["x"], ["y"], blocksize=2, mode="DCR", dtype=dtype
    ).forward(_tensor(depth, dtype))["tensor"].data
    np.testing.assert_array_equal(depth_to_space, values.reshape(1, 1, 2, 2))

    resize_input = values.reshape(1, 1, 1, 4)
    sizes = np.array([1, 1, 1, 8], dtype=np.int64)
    resized = Resize(
        ["x", "", "", "sizes"],
        ["y"],
        mode="nearest",
        coord_mode="asymmetric",
        nearest_mode="floor",
        dtype=dtype,
    ).forward(_tensor(resize_input, dtype), None, None, _tensor(sizes, "int64"))["tensor"].data
    np.testing.assert_array_equal(resized, np.repeat(resize_input, 2, axis=3))


def test_native_layout_movement_preserves_float32_payload_bits():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    bits = np.array([0x7FC12345, 0xFFC54321, 0x80000000, 0x3F800000], dtype=np.uint32)
    values = bits.view(np.float32).reshape(2, 2)

    transposed = Transpose(["x"], ["y"], perm=[1, 0], dtype="float32").forward(
        _tensor(values, "float32")
    )["tensor"].data
    np.testing.assert_array_equal(transposed.view(np.uint32), bits.reshape(2, 2).T)

    resize_input = values.reshape(1, 1, 1, 4)
    sizes = np.array([1, 1, 1, 8], dtype=np.int64)
    resized = Resize(
        ["x", "", "", "sizes"],
        ["y"],
        mode="nearest",
        coord_mode="asymmetric",
        nearest_mode="floor",
        dtype="float32",
    ).forward(_tensor(resize_input, "float32"), None, None, _tensor(sizes, "int64"))["tensor"].data
    expected_bits = np.repeat(bits.reshape(1, 1, 1, 4), 2, axis=3)
    np.testing.assert_array_equal(resized.view(np.uint32), expected_bits)


@pytest.mark.parametrize(
    "dtype,proto",
    [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)],
)
def test_strict_imported_transpose_preserves_large_integer_payloads(tmp_path, dtype, proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = _large_values(dtype).reshape(2, 2)
    graph = helper.make_graph(
        [helper.make_node("Transpose", ["x"], ["y"], perm=[1, 0])],
        "transpose_integer_payload",
        [helper.make_tensor_value_info("x", proto, [2, 2])],
        [helper.make_tensor_value_info("y", proto, [2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    model_path = tmp_path / f"transpose_{dtype}.onnx"
    onnx.save(model, model_path)

    actual = Graph(ONNXImport(str(model_path), strict=True), ["x"], ["y"]).forward(
        _tensor(data, dtype)
    )
    np.testing.assert_array_equal(actual.data, data.T)


@pytest.mark.parametrize(
    "dtype,proto",
    [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)],
)
def test_strict_imported_nearest_resize_preserves_large_integer_payloads(tmp_path, dtype, proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = _large_values(dtype).reshape(1, 1, 1, 4)
    sizes = numpy_helper.from_array(np.array([1, 1, 1, 8], dtype=np.int64), "sizes")
    graph = helper.make_graph(
        [
            helper.make_node(
                "Resize",
                ["x", "", "", "sizes"],
                ["y"],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
        ],
        "resize_integer_payload",
        [helper.make_tensor_value_info("x", proto, [1, 1, 1, 4])],
        [helper.make_tensor_value_info("y", proto, [1, 1, 1, 8])],
        initializer=[sizes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    model_path = tmp_path / f"resize_{dtype}.onnx"
    onnx.save(model, model_path)

    actual = Graph(ONNXImport(str(model_path), strict=True), ["x"], ["y"]).forward(
        _tensor(data, dtype)
    )
    np.testing.assert_array_equal(actual.data, np.repeat(data, 2, axis=3))


@pytest.mark.parametrize(
    "dtype,proto",
    [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)],
)
def test_strict_imported_space_to_depth_preserves_large_integer_payloads(tmp_path, dtype, proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = _large_values(dtype).reshape(1, 1, 2, 2)
    graph = helper.make_graph(
        [helper.make_node("SpaceToDepth", ["x"], ["y"], blocksize=2)],
        "space_to_depth_integer_payload",
        [helper.make_tensor_value_info("x", proto, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", proto, [1, 4, 1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    model_path = tmp_path / f"space_to_depth_{dtype}.onnx"
    onnx.save(model, model_path)

    actual = Graph(ONNXImport(str(model_path), strict=True), ["x"], ["y"]).forward(
        _tensor(data, dtype)
    )
    np.testing.assert_array_equal(actual.data, _large_values(dtype).reshape(1, 4, 1, 1))


def _scatter_model(path, reduction, dtype, proto):
    graph = helper.make_graph(
        [
            helper.make_node(
                "ScatterElements",
                ["data", "indices", "updates"],
                ["output"],
                axis=0,
                reduction=reduction,
            )
        ],
        f"scatter_elements_{reduction}",
        [
            helper.make_tensor_value_info("data", proto, [2]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("updates", proto, [2]),
        ],
        [helper.make_tensor_value_info("output", proto, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return Graph(ONNXImport(str(path), strict=True), ["data", "indices", "updates"], ["output"])


@pytest.mark.parametrize("reduction,expected", [("max", [10, 30]), ("min", [5, 20])])
@pytest.mark.parametrize(
    "dtype,proto",
    [("int64", TensorProto.INT64), ("float32", TensorProto.FLOAT)],
)
def test_strict_imported_scatter_elements_opset18_max_min(tmp_path, reduction, expected, dtype, proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    np_dtype = np.dtype(dtype)
    data = np.array([10, 20], dtype=np_dtype)
    indices = np.array([0, 1], dtype=np.int64)
    updates = np.array([5, 30], dtype=np_dtype)
    runtime = _scatter_model(tmp_path / f"scatter_{dtype}_{reduction}.onnx", reduction, dtype, proto)
    actual = runtime.forward(
        _tensor(data, dtype), _tensor(indices, "int64"), _tensor(updates, dtype)
    )
    np.testing.assert_array_equal(actual.data, np.array(expected, dtype=np_dtype))


@pytest.mark.parametrize("reduction,expected", [("max", [2**63 + 2, 2**64 - 1]), ("min", [2**63 + 1, 2**64 - 2])])
def test_scatter_elements_max_min_compare_uint64_exactly(reduction, expected):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([2**63 + 2, 2**64 - 1], dtype=np.uint64)
    updates = np.array([2**63 + 1, 2**64 - 2], dtype=np.uint64)
    indices = np.array([0, 1], dtype=np.int64)
    actual = ScatterElements(
        ["data", "indices", "updates"],
        ["output"],
        axis=0,
        reduction=reduction,
        dtype="uint64",
        version="18",
    ).forward(_tensor(data, "uint64"), _tensor(indices, "int64"), _tensor(updates, "uint64"))["tensor"]
    np.testing.assert_array_equal(actual.data, np.array(expected, dtype=np.uint64))


@pytest.mark.parametrize("reduction,expected", [("max", [9]), ("min", [2])])
def test_scatter_elements_max_min_reduce_duplicate_indices(reduction, expected):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([5], dtype=np.int64)
    indices = np.array([0, 0, 0], dtype=np.int64)
    updates = np.array([2, 9, 4], dtype=np.int64)
    actual = ScatterElements(
        ["data", "indices", "updates"],
        ["output"],
        axis=0,
        reduction=reduction,
        dtype="int64",
        version="18",
    ).forward(_tensor(data, "int64"), _tensor(indices, "int64"), _tensor(updates, "int64"))["tensor"]
    np.testing.assert_array_equal(actual.data, np.array(expected, dtype=np.int64))


@pytest.mark.parametrize("reduction", ["max", "min"])
def test_scatter_elements_max_min_require_opset18(reduction):
    with pytest.raises(ValueError, match="requires ONNX opset 18"):
        ScatterElements(
            ["data", "indices", "updates"],
            ["output"],
            reduction=reduction,
            dtype="float32",
            version="17",
        )


@pytest.mark.parametrize("reduction", ["max", "min", "unsupported"])
def test_scatter_nd_direct_api_rejects_unsupported_reduction(reduction):
    with pytest.raises(ValueError, match="ScatterND reduction"):
        ScatterND(
            ["data", "indices", "updates"],
            ["output"],
            reduction=reduction,
            dtype="int64",
            version="18",
        )
