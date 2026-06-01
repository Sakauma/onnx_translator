import numpy as np
import pytest

from nn import Graph, Tensor, Tensor_


class PassThroughOp:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    def forward(self, x):
        return {"tensor": x, "parameters": None}

    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}


class AddOneOp:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    def forward(self, x):
        return {
            "tensor": Tensor(*x.size, dtype=x.dtype, data=x.data + 1),
            "parameters": None,
        }

    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}


class SplitOp:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    def forward(self, x):
        doubled = Tensor(*x.size, dtype=x.dtype, data=x.data * 2)
        return {"tensor": (x, doubled), "parameters": None}

    def forward_(self, x):
        return {
            "tensor": (
                Tensor_(*x.size, dtype=x.dtype),
                Tensor_(*x.size, dtype=x.dtype),
            ),
            "parameters": None,
        }


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


def test_graph_forward_validates_input_count():
    graph = Graph(
        ops=[PassThroughOp(["input"], ["output"])],
        input_name=["input"],
    )

    with pytest.raises(ValueError, match="expects 1 inputs"):
        graph.forward()


def test_graph_forward_raises_for_missing_declared_output():
    graph = Graph(
        ops=[PassThroughOp(["input"], ["output"])],
        input_name=["input"],
        output_name=["missing"],
    )
    x = Tensor(1, dtype="float32", data=np.array([1.0], dtype=np.float32))

    with pytest.raises(KeyError, match="missing"):
        graph.forward(x)
