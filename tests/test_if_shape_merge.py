# /**
#   ******************************************************************************
#   * @file        test_if_shape_merge.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖 If 分支输出元数据的保守合并语义。
#   * @details     2026.09.21  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np
import pytest
from onnx import TensorProto, helper

from conftest import _disable_c_backend
from nn import Graph, Tensor, Tensor_
from nn.Operators import ADD, Constant, Identity, If


def _branch(name, outputs):
    nodes = []
    value_infos = []
    for index, (dtype, declared_shape, data) in enumerate(outputs):
        output_name = f"out_{index}"
        array = np.asarray(data)
        tensor = helper.make_tensor(
            f"value_{index}", dtype, list(array.shape), array.reshape(-1).tolist()
        )
        nodes.append(helper.make_node("Constant", [], [output_name], value=tensor))
        value_infos.append(
            helper.make_tensor_value_info(output_name, dtype, declared_shape)
        )
    return helper.make_graph(nodes, name, [], value_infos)


def _cond(value):
    return Tensor(dtype="bool", data=np.array(value, dtype=np.bool_))


def test_if_merges_equal_rank_dimensions_and_preserves_equal_shape(monkeypatch):
    _disable_c_backend(monkeypatch)
    same_then = _branch("same_then", [(TensorProto.FLOAT, [2, 3], np.ones((2, 3), np.float32))])
    same_else = _branch("same_else", [(TensorProto.FLOAT, [2, 3], np.zeros((2, 3), np.float32))])
    same = If(["cond"], ["y"], same_then, same_else).forward_(Tensor_(dtype="bool"))["tensor"]
    assert same.size == (2, 3)
    assert same.data_size == 6

    then_branch = _branch("then", [(TensorProto.FLOAT, [2], np.ones(2, np.float32))])
    else_branch = _branch("else", [(TensorProto.FLOAT, [3], np.ones(3, np.float32))])
    merged = If(["cond"], ["y"], then_branch, else_branch).forward_(Tensor_(dtype="bool"))["tensor"]
    assert merged.size == (None,)
    assert merged.dtype == "float32"
    assert merged.data_size is None
    assert "size=(None,)" in repr(merged)
    with pytest.raises(ValueError, match="unknown dimension"):
        merged.require_concrete_shape("AllocateOutput")


def test_if_distinguishes_scalar_shape_from_unknown_rank(monkeypatch):
    _disable_c_backend(monkeypatch)
    scalar_then = _branch(
        "scalar_then", [(TensorProto.FLOAT, [], np.array(1.0, np.float32))]
    )
    scalar_else = _branch(
        "scalar_else", [(TensorProto.FLOAT, [], np.array(2.0, np.float32))]
    )
    scalar = If(
        ["cond"], ["y"], scalar_then, scalar_else
    ).forward_(Tensor_(dtype="bool"))["tensor"]
    assert scalar.size == ()
    assert scalar.data_size == 1

    unknown_rank_else = _branch(
        "unknown_rank_else",
        [(TensorProto.FLOAT, None, np.array(2.0, np.float32))],
    )
    unknown_rank = If(
        ["cond"], ["y"], scalar_then, unknown_rank_else
    ).forward_(Tensor_(dtype="bool"))["tensor"]
    assert unknown_rank.size is None
    assert unknown_rank.data_size is None


def test_if_unknown_dimension_is_preserved_through_identity(monkeypatch):
    _disable_c_backend(monkeypatch)
    then_branch = _branch(
        "then_unknown", [(TensorProto.FLOAT, [None, 3], np.ones((2, 3), np.float32))]
    )
    else_branch = _branch(
        "else_known", [(TensorProto.FLOAT, [4, 3], np.ones((4, 3), np.float32))]
    )
    graph = Graph(
        [
            If(["cond"], ["branch_y"], then_branch, else_branch),
            Identity(["branch_y"], ["y"], dtype="float32"),
        ],
        ["cond"],
        ["y"],
    )
    result = graph.forward_(Tensor_(dtype="bool"))
    assert result.size == (None, 3)
    assert result.data_size is None


def test_if_rank_mismatch_has_explicit_unknown_rank_and_identity_preserves_it(monkeypatch):
    _disable_c_backend(monkeypatch)
    then_branch = _branch("rank_one", [(TensorProto.FLOAT, [2], np.ones(2, np.float32))])
    else_branch = _branch("rank_two", [(TensorProto.FLOAT, [1, 2], np.ones((1, 2), np.float32))])
    graph = Graph(
        [
            If(["cond"], ["branch_y"], then_branch, else_branch),
            Identity(["branch_y"], ["y"], dtype="float32"),
        ],
        ["cond"],
        ["y"],
    )
    result = graph.forward_(Tensor_(dtype="bool"))
    assert result.size is None
    assert result.data_size is None
    assert result.dtype == "float32"
    assert "size=None" in repr(result)
    with pytest.raises(ValueError, match="requires a statically known input rank"):
        result.require_known_rank("DownstreamShapeOp")

    requires_shape = Graph(
        [
            Constant([], ["bias"], value=np.array([1.0], dtype=np.float32)),
            If(["cond"], ["branch_y"], then_branch, else_branch),
            ADD(["branch_y", "bias"], ["y"], dtype="float32"),
        ],
        ["cond"],
        ["y"],
    )
    with pytest.raises(ValueError, match="ADD.forward_ requires concrete input shape"):
        requires_shape.forward_(Tensor_(dtype="bool"))


def test_if_merges_each_output_and_rejects_dtype_mismatch(monkeypatch):
    _disable_c_backend(monkeypatch)
    then_branch = _branch(
        "multi_then",
        [
            (TensorProto.FLOAT, [2], np.ones(2, np.float32)),
            (TensorProto.INT64, [1, 4], np.ones((1, 4), np.int64)),
        ],
    )
    else_branch = _branch(
        "multi_else",
        [
            (TensorProto.FLOAT, [5], np.ones(5, np.float32)),
            (TensorProto.INT64, [3, 4], np.ones((3, 4), np.int64)),
        ],
    )
    first, second = If(
        ["cond"], ["a", "b"], then_branch, else_branch
    ).forward_(Tensor_(dtype="bool"))["tensor"]
    assert first.size == (None,)
    assert first.dtype == "float32"
    assert second.size == (None, 4)
    assert second.dtype == "int64"

    dtype_then = _branch(
        "dtype_then", [(TensorProto.FLOAT, [2], np.ones(2, np.float32))]
    )
    mismatched = _branch(
        "dtype_else", [(TensorProto.INT64, [2], np.ones(2, np.int64))]
    )
    with pytest.raises(TypeError, match="branch dtype mismatch"):
        If(["cond"], ["y"], dtype_then, mismatched).forward_(Tensor_(dtype="bool"))


def test_if_real_execution_still_selects_each_branch(monkeypatch):
    _disable_c_backend(monkeypatch)
    then_branch = _branch(
        "real_then", [(TensorProto.FLOAT, [2], np.array([1.0, 2.0], np.float32))]
    )
    else_branch = _branch(
        "real_else", [(TensorProto.FLOAT, [3], np.array([3.0, 4.0, 5.0], np.float32))]
    )
    op = If(["cond"], ["y"], then_branch, else_branch)
    assert op.forward(_cond(True))["tensor"].size == (2,)
    assert op.forward(_cond(False))["tensor"].size == (3,)

    graph = Graph(
        [
            If(["cond"], ["branch_y"], then_branch, else_branch),
            Identity(["branch_y"], ["y"], dtype="float32"),
        ],
        ["cond"],
        ["y"],
    )
    assert graph.forward(_cond(True)).size == (2,)
    assert graph.forward(_cond(False)).size == (3,)
