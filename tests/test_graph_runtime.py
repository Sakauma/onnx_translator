# /**
#   ******************************************************************************
#   * @file        test_graph_runtime.py
#   * @author      Egor Izmaylov
#   * @brief       验证 Graph 运行时的输入校验、输出收集、多输出和重复执行行为。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nn import Graph, Tensor, Tensor_
from tools.commands import verify_graph


class PassThroughOp:
    # 初始化 `PassThroughOp` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    # 执行 `PassThroughOp` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": x, "parameters": None}

    # 执行 `PassThroughOp` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}


class AddOneOp:
    # 初始化 `AddOneOp` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    # 执行 `AddOneOp` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {
            "tensor": Tensor(*x.size, dtype=x.dtype, data=x.data + 1),
            "parameters": None,
        }

    # 执行 `AddOneOp` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}


class SplitOp:
    # 初始化 `SplitOp` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    # 执行 `SplitOp` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        doubled = Tensor(*x.size, dtype=x.dtype, data=x.data * 2)
        return {"tensor": (x, doubled), "parameters": None}

    # 执行 `SplitOp` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {
            "tensor": (
                Tensor_(*x.size, dtype=x.dtype),
                Tensor_(*x.size, dtype=x.dtype),
            ),
            "parameters": None,
        }


# 验证 `test_graph_forward_returns_inferred_output_and_is_repeatable` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_graph_forward_returns_inferred_output_and_is_repeatable():
    graph = Graph(
        ops=[
            PassThroughOp(["input"], ["hidden"]),
            AddOneOp(["hidden"], ["output"]),
        ],
        input_name=["input"],
    )
    x = Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32))

    first = graph.forward(x)
    second = graph.forward(x)

    np.testing.assert_array_equal(first.data, np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(second.data, first.data)


# 验证 `test_graph_forward_shape_inference_returns_declared_output` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_graph_forward_shape_inference_returns_declared_output():
    graph = Graph(
        ops=[
            PassThroughOp(["input"], ["hidden"]),
            AddOneOp(["hidden"], ["output"]),
        ],
        input_name=["input"],
        output_name=["output"],
    )
    out = graph.forward_(Tensor_(3, dtype="float32"))

    assert out.size == (3,)
    assert out.dtype == "float32"


# 验证 `test_graph_forward_infers_multiple_terminal_outputs` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_graph_forward_infers_multiple_terminal_outputs():
    graph = Graph(
        ops=[
            SplitOp(["input"], ["left", "right"]),
            AddOneOp(["left"], ["left_out"]),
        ],
        input_name=["input"],
    )
    x = Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32))

    right, left_out = graph.forward(x)

    np.testing.assert_array_equal(right.data, np.array([2.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(left_out.data, np.array([2.0, 3.0], dtype=np.float32))


# 验证 `test_graph_forward_validates_input_count` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_graph_forward_validates_input_count():
    graph = Graph(
        ops=[PassThroughOp(["input"], ["output"])],
        input_name=["input"],
    )

    with pytest.raises(ValueError, match="expects 1 inputs"):
        graph.forward()


# 验证 `test_graph_forward_raises_for_missing_declared_output` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_graph_forward_raises_for_missing_declared_output():
    graph = Graph(
        ops=[PassThroughOp(["input"], ["output"])],
        input_name=["input"],
        output_name=["missing"],
    )
    x = Tensor(1, dtype="float32", data=np.array([1.0], dtype=np.float32))

    with pytest.raises(KeyError, match="missing"):
        graph.forward(x)


def test_verify_graph_uses_declared_output_order_and_ignores_extra_terminal(tmp_path, monkeypatch):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    final = helper.make_tensor_value_info("final", TensorProto.FLOAT, [2])
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["hidden"]),
            helper.make_node("Identity", ["hidden"], ["final"]),
            helper.make_node("Neg", ["x"], ["extra_terminal"]),
        ],
        "declared_outputs",
        [x],
        [final, hidden],
    )
    model_path = tmp_path / "declared_outputs.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)
    monkeypatch.setattr(verify_graph, "GraphGenerate", lambda graph, task_name: None)

    assert verify_graph.run_verification(str(model_path), "declared_outputs", clean=False) == 0


def test_declared_output_validation_rejects_dtype_rank_and_known_dimension_mismatch():
    declaration = [("out", TensorProto.FLOAT, (None, 3))]
    # Symbolic/unknown first dimension is deliberately unconstrained.
    verify_graph._validate_declared_outputs(declaration, Tensor_(7, 3, dtype="float32"))

    with pytest.raises(TypeError, match="dtype mismatch"):
        verify_graph._validate_declared_outputs(declaration, Tensor_(7, 3, dtype="int32"))
    with pytest.raises(ValueError, match="rank mismatch"):
        verify_graph._validate_declared_outputs(declaration, Tensor_(3, dtype="float32"))
    with pytest.raises(ValueError, match="dimension 1 mismatch"):
        verify_graph._validate_declared_outputs(declaration, Tensor_(7, 4, dtype="float32"))


def test_declared_output_validation_rejects_output_count_mismatch():
    declarations = [
        ("first", TensorProto.FLOAT, (1,)),
        ("second", TensorProto.FLOAT, (1,)),
    ]
    with pytest.raises(ValueError, match="output count mismatch"):
        verify_graph._validate_declared_outputs(declarations, Tensor_(1, dtype="float32"))
