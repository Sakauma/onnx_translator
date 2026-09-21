# /**
#   ******************************************************************************
#   * @file        test_gather_integer_preservation.py
#   * @author      Egor Izmaylov
#   * @brief       验证 Gather 算子族精确保留所选元素的底层数值。
#   * @details     2026.09.21  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import os

import numpy as np
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

import nn
from nn import Graph, Tensor
from nn.Operators import Gather, GatherElements, GatherND


def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


def _reference(op_type, data, indices, data_proto, attrs, output_shape):
    graph = helper.make_graph(
        [helper.make_node(op_type, ["data", "indices"], ["output"], **attrs)],
        f"{op_type}_integer_preservation",
        [
            helper.make_tensor_value_info("data", data_proto, list(data.shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
        ],
        [helper.make_tensor_value_info("output", data_proto, list(output_shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    return ReferenceEvaluator(model).run(None, {"data": data, "indices": indices})[0]


def _integer_data(dtype):
    if dtype == "int64":
        return np.array(
            [
                [
                    [np.iinfo(np.int64).min, np.iinfo(np.int64).max],
                    [2**53 + 1, -(2**53 + 1)],
                    [2**53 + 3, -(2**53 + 3)],
                ],
                [
                    [np.iinfo(np.int64).max - 2, np.iinfo(np.int64).min + 2],
                    [-1, 0],
                    [2**53 + 5, -(2**53 + 5)],
                ],
            ],
            dtype=np.int64,
        )

    return np.array(
        [
            [[0, np.iinfo(np.uint64).max], [2**53 + 1, 2**53 + 3], [2**63 + 1, 2**63 + 3]],
            [[np.iinfo(np.uint64).max - 1, 1], [2**53 - 1, 2**53 + 5], [2**63, 2**64 - 2]],
        ],
        dtype=np.uint64,
    )


def _assert_exact(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype,data_proto",
    [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)],
)
def test_gather_graph_preserves_large_integer_payloads(dtype, data_proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = _integer_data(dtype)
    indices = np.array([[-1, 0], [1, -2]], dtype=np.int64)
    expected = _reference(
        "Gather",
        data,
        indices,
        data_proto,
        {"axis": 1},
        (2, 2, 2, 2),
    )
    op = Gather(["data", "indices"], ["output"], axis=1, dtype=dtype)
    actual = Graph([op], ["data", "indices"], ["output"]).forward(
        _tensor(data, dtype),
        _tensor(indices, "int64"),
    )

    _assert_exact(actual.data, expected)


@pytest.mark.parametrize(
    "dtype,data_proto",
    [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)],
)
def test_gather_elements_preserves_large_integer_payloads(dtype, data_proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = _integer_data(dtype)
    indices = np.array(
        [
            [[-1, -1], [0, 0], [1, 1]],
            [[1, 1], [-1, -1], [0, 0]],
        ],
        dtype=np.int64,
    )
    expected = _reference(
        "GatherElements",
        data,
        indices,
        data_proto,
        {"axis": 1},
        indices.shape,
    )
    actual = GatherElements(["data", "indices"], ["output"], axis=1, dtype=dtype).forward(
        _tensor(data, dtype),
        _tensor(indices, "int64"),
    )["tensor"]

    _assert_exact(actual.data, expected)


@pytest.mark.parametrize(
    "dtype,data_proto",
    [("int64", TensorProto.INT64), ("uint64", TensorProto.UINT64)],
)
def test_gather_nd_preserves_large_integer_payloads(dtype, data_proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = _integer_data(dtype)
    indices = np.array(
        [
            [[0, 0], [0, -1], [0, -2]],
            [[-1, 0], [-1, -1], [-1, -2]],
        ],
        dtype=np.int64,
    )
    expected = _reference(
        "GatherND",
        data,
        indices,
        data_proto,
        {"batch_dims": 0},
        (2, 3, 2),
    )
    actual = GatherND(["data", "indices"], ["output"], batch_dims=0, dtype=dtype).forward(
        _tensor(data, dtype),
        _tensor(indices, "int64"),
    )["tensor"]

    _assert_exact(actual.data, expected)


@pytest.mark.parametrize("op_type", ["Gather", "GatherElements", "GatherND"])
def test_gather_family_preserves_float32_payload_bits(op_type):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array(
        [
            0x00000000,
            0x80000000,
            0x7FC12345,
            0xFFC54321,
            0x7F800000,
            0xFF800000,
            0x3F800000,
            0xBF800000,
            0x40490FDB,
            0xC0490FDB,
            0x00800000,
            0x80800000,
        ],
        dtype=np.uint32,
    ).view(np.float32).reshape(2, 3, 2)

    if op_type == "Gather":
        indices = np.array([[-1, 0], [1, -2]], dtype=np.int64)
        attrs = {"axis": 1}
        output_shape = (2, 2, 2, 2)
        op = Gather(["data", "indices"], ["output"], axis=1, dtype="float32")
        actual = Graph([op], ["data", "indices"], ["output"]).forward(
            _tensor(data, "float32"),
            _tensor(indices, "int64"),
        ).data
    elif op_type == "GatherElements":
        indices = np.array(
            [
                [[-1, -1], [0, 0], [1, 1]],
                [[1, 1], [-1, -1], [0, 0]],
            ],
            dtype=np.int64,
        )
        attrs = {"axis": 1}
        output_shape = indices.shape
        actual = GatherElements(["data", "indices"], ["output"], axis=1, dtype="float32").forward(
            _tensor(data, "float32"),
            _tensor(indices, "int64"),
        )["tensor"].data
    else:
        indices = np.array(
            [
                [[0, 0], [0, -1], [0, -2]],
                [[-1, 0], [-1, -1], [-1, -2]],
            ],
            dtype=np.int64,
        )
        attrs = {"batch_dims": 0}
        output_shape = (2, 3, 2)
        actual = GatherND(["data", "indices"], ["output"], batch_dims=0, dtype="float32").forward(
            _tensor(data, "float32"),
            _tensor(indices, "int64"),
        )["tensor"].data

    expected = _reference(op_type, data, indices, TensorProto.FLOAT, attrs, output_shape)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32))
