import numpy as np
import pytest

from nn import Graph, Tensor, Tensor_


class PassThroughOp:
    # Egor Izmaylov: Function `PassThroughOp.__init__` initializes PassThroughOp, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    # Egor Izmaylov: Function `PassThroughOp.forward` executes the concrete runtime path for PassThroughOp, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": x, "parameters": None}

    # Egor Izmaylov: Function `PassThroughOp.forward_` performs shape-only inference for PassThroughOp, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}


class AddOneOp:
    # Egor Izmaylov: Function `AddOneOp.__init__` initializes AddOneOp, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    # Egor Izmaylov: Function `AddOneOp.forward` executes the concrete runtime path for AddOneOp, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {
            "tensor": Tensor(*x.size, dtype=x.dtype, data=x.data + 1),
            "parameters": None,
        }

    # Egor Izmaylov: Function `AddOneOp.forward_` performs shape-only inference for AddOneOp, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}


class SplitOp:
    # Egor Izmaylov: Function `SplitOp.__init__` initializes SplitOp, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.name = None

    # Egor Izmaylov: Function `SplitOp.forward` executes the concrete runtime path for SplitOp, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        doubled = Tensor(*x.size, dtype=x.dtype, data=x.data * 2)
        return {"tensor": (x, doubled), "parameters": None}

    # Egor Izmaylov: Function `SplitOp.forward_` performs shape-only inference for SplitOp, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {
            "tensor": (
                Tensor_(*x.size, dtype=x.dtype),
                Tensor_(*x.size, dtype=x.dtype),
            ),
            "parameters": None,
        }


# Egor Izmaylov: Function `test_graph_forward_returns_inferred_output_and_is_repeatable` locks down the test graph forward returns inferred output and is repeatable behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
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


# Egor Izmaylov: Function `test_graph_forward_shape_inference_returns_declared_output` locks down the test graph forward shape inference returns declared output behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
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


# Egor Izmaylov: Function `test_graph_forward_infers_multiple_terminal_outputs` locks down the test graph forward infers multiple terminal outputs behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
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


# Egor Izmaylov: Function `test_graph_forward_validates_input_count` locks down the test graph forward validates input count behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
def test_graph_forward_validates_input_count():
    graph = Graph(
        ops=[PassThroughOp(["input"], ["output"])],
        input_name=["input"],
    )

    with pytest.raises(ValueError, match="expects 1 inputs"):
        graph.forward()


# Egor Izmaylov: Function `test_graph_forward_raises_for_missing_declared_output` locks down the test graph forward raises for missing declared output behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
def test_graph_forward_raises_for_missing_declared_output():
    graph = Graph(
        ops=[PassThroughOp(["input"], ["output"])],
        input_name=["input"],
        output_name=["missing"],
    )
    x = Tensor(1, dtype="float32", data=np.array([1.0], dtype=np.float32))

    with pytest.raises(KeyError, match="missing"):
        graph.forward(x)
