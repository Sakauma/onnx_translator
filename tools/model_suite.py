# /**
#   ******************************************************************************
#   * @file        model_suite.py
#   * @author      Egor Izmaylov
#   * @brief       Generates and verifies representative ONNX model smoke tests.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nn
import nn.ModelInitParas
from nn import Graph, Tensor, Tensor_
from nn.ONNXImport import ONNXImport
from nn.importer.context import GenericNode


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    builder: Callable[[Path], Path]
    expected_outputs: dict[str, tuple[int, ...]]
    numeric_rtol: float = 1e-4
    numeric_atol: float = 1e-4


def _float_data(shape: tuple[int, ...], scale: float = 0.01) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return (values * scale).astype(np.float32)


def _init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(value), name=name)


def _value(name: str, elem_type: int, shape: tuple[int, ...]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def _save_model(
    output_path: Path,
    graph_name: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="onnx_translator_model_suite")
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def _build_vision_cnn(output_dir: Path) -> Path:
    initializers = [
        _init("conv1_w", _float_data((4, 3, 3, 3))),
        _init("conv1_b", _float_data((4,), 0.001)),
        _init("bn_scale", np.ones((4,), dtype=np.float32)),
        _init("bn_bias", np.zeros((4,), dtype=np.float32)),
        _init("bn_mean", np.zeros((4,), dtype=np.float32)),
        _init("bn_var", np.ones((4,), dtype=np.float32)),
        _init("conv2_w", _float_data((8, 4, 3, 3), 0.005)),
        _init("conv2_b", _float_data((8,), 0.001)),
        _init("fc_w", _float_data((8, 10), 0.002)),
        _init("fc_b", _float_data((10,), 0.001)),
    ]
    nodes = [
        helper.make_node("Conv", ["image", "conv1_w", "conv1_b"], ["conv1"], name="suite_conv1", pads=[1, 1, 1, 1]),
        helper.make_node(
            "BatchNormalization",
            ["conv1", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["bn1"],
            name="suite_bn1",
            epsilon=1e-5,
        ),
        helper.make_node("Relu", ["bn1"], ["relu1"], name="suite_relu1"),
        helper.make_node("MaxPool", ["relu1"], ["pool1"], name="suite_pool1", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["pool1", "conv2_w", "conv2_b"], ["conv2"], name="suite_conv2", pads=[1, 1, 1, 1]),
        helper.make_node("GlobalAveragePool", ["conv2"], ["gap"], name="suite_gap"),
        helper.make_node("Flatten", ["gap"], ["flat"], name="suite_flatten", axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"], name="suite_gemm"),
        helper.make_node("Softmax", ["logits"], ["probs"], name="suite_softmax", axis=1),
    ]
    return _save_model(
        output_dir / "vision_cnn.onnx",
        "vision_cnn",
        nodes,
        [_value("image", TensorProto.FLOAT, (1, 3, 16, 16))],
        [_value("probs", TensorProto.FLOAT, (1, 10))],
        initializers,
    )


def _build_transformer_block(output_dir: Path) -> Path:
    initializers = [
        _init("wq", _float_data((8, 8), 0.01)),
        _init("wk", _float_data((8, 8), 0.008)),
        _init("wv", _float_data((8, 8), 0.006)),
        _init("wo", _float_data((8, 8), 0.004)),
        _init("attn_scale", np.array(1.0 / np.sqrt(8.0), dtype=np.float32)),
        _init("ln_scale", np.ones((8,), dtype=np.float32)),
        _init("ln_bias", np.zeros((8,), dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("MatMul", ["tokens", "wq"], ["q"], name="suite_q"),
        helper.make_node("MatMul", ["tokens", "wk"], ["k"], name="suite_k"),
        helper.make_node("MatMul", ["tokens", "wv"], ["v"], name="suite_v"),
        helper.make_node("Transpose", ["k"], ["kt"], name="suite_k_transpose", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["scores"], name="suite_scores"),
        helper.make_node("Mul", ["scores", "attn_scale"], ["scaled_scores"], name="suite_scale"),
        helper.make_node("Softmax", ["scaled_scores"], ["attention"], name="suite_attention", axis=-1),
        helper.make_node("MatMul", ["attention", "v"], ["context"], name="suite_context"),
        helper.make_node("MatMul", ["context", "wo"], ["projected"], name="suite_project"),
        helper.make_node("Add", ["projected", "tokens"], ["residual"], name="suite_residual"),
        helper.make_node(
            "LayerNormalization",
            ["residual", "ln_scale", "ln_bias"],
            ["normalized"],
            name="suite_layer_norm",
            axis=-1,
            epsilon=1e-5,
        ),
    ]
    return _save_model(
        output_dir / "transformer_block.onnx",
        "transformer_block",
        nodes,
        [_value("tokens", TensorProto.FLOAT, (2, 4, 8))],
        [_value("normalized", TensorProto.FLOAT, (2, 4, 8))],
        initializers,
    )


def _build_embedding_mlp(output_dir: Path) -> Path:
    initializers = [
        _init("embedding", _float_data((16, 8), 0.01)),
        _init("fc1_w", _float_data((12, 6), 0.01)),
        _init("fc1_b", np.zeros((6,), dtype=np.float32)),
        _init("fc2_w", _float_data((6, 2), 0.01)),
        _init("fc2_b", np.zeros((2,), dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("Gather", ["embedding", "item_ids"], ["embedded"], name="suite_gather", axis=0),
        helper.make_node("ReduceMean", ["embedded"], ["pooled"], name="suite_pool_embedding", axes=[1], keepdims=0),
        helper.make_node("Concat", ["pooled", "dense"], ["features"], name="suite_concat", axis=1),
        helper.make_node("Gemm", ["features", "fc1_w", "fc1_b"], ["hidden"], name="suite_fc1"),
        helper.make_node("Relu", ["hidden"], ["activated"], name="suite_relu"),
        helper.make_node("Gemm", ["activated", "fc2_w", "fc2_b"], ["logits"], name="suite_fc2"),
        helper.make_node("Sigmoid", ["logits"], ["scores"], name="suite_sigmoid"),
    ]
    return _save_model(
        output_dir / "embedding_mlp.onnx",
        "embedding_mlp",
        nodes,
        [
            _value("item_ids", TensorProto.INT64, (2, 3)),
            _value("dense", TensorProto.FLOAT, (2, 4)),
        ],
        [_value("scores", TensorProto.FLOAT, (2, 2))],
        initializers,
    )


MODEL_SPECS = [
    ModelSpec(
        "vision_cnn",
        "CNN classifier block with normalization, pooling, GEMM, and Softmax.",
        _build_vision_cnn,
        {"probs": (1, 10)},
    ),
    ModelSpec(
        "transformer_block",
        "Tiny attention block with MatMul, Softmax, residual Add, and LayerNormalization.",
        _build_transformer_block,
        {"normalized": (2, 4, 8)},
    ),
    ModelSpec(
        "embedding_mlp",
        "Embedding and dense-feature MLP with Gather, ReduceMean, Concat, GEMM, and Sigmoid.",
        _build_embedding_mlp,
        {"scores": (2, 2)},
    ),
]


def generate_model_suite(output_dir: Path) -> dict[str, Path]:
    return {spec.name: spec.builder(output_dir) for spec in MODEL_SPECS}


def _as_tuple(output: object) -> tuple[object, ...]:
    if isinstance(output, tuple):
        return output
    if isinstance(output, list):
        return tuple(output)
    return (output,)


def _input_data(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    size = int(np.prod(shape)) if shape else 1
    np_dtype = nn.DTYPE_TO_NUMPY.get(dtype, np.float32)
    if dtype == "bool":
        return (np.arange(size).reshape(shape) % 2 == 0).astype(np_dtype)
    if dtype.startswith("int") or dtype.startswith("uint"):
        return (np.arange(size, dtype=np.int64).reshape(shape) % 4).astype(np_dtype)
    values = np.linspace(-0.5, 0.5, num=size, dtype=np.float32).reshape(shape)
    return values.astype(np_dtype)


def _prepare_inputs(initial_inputs: list[str], initial_tensors: list[Tensor]) -> tuple[dict[str, np.ndarray], list[Tensor], list[Tensor_]]:
    feeds = {}
    runtime_inputs = []
    placeholders = []
    for name, tensor in zip(initial_inputs, initial_tensors):
        shape = tuple(int(dim) for dim in tensor.size)
        data = _input_data(shape, tensor.dtype)
        feeds[name] = data
        runtime_inputs.append(Tensor(*shape, dtype=tensor.dtype, data=data))
        placeholders.append(Tensor_(*shape, dtype=tensor.dtype))
    return feeds, runtime_inputs, placeholders


def _numeric_diff(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    if reference.size == 0:
        return 0.0, 0.0
    ref = reference.astype(np.float64, copy=False)
    got = actual.astype(np.float64, copy=False)
    abs_diff = np.abs(got - ref)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.abs(ref), 1e-12)
    max_rel = float(np.max(abs_diff / denom))
    return max_abs, max_rel


def _compare_numeric_outputs(
    spec: ModelSpec,
    output_names: list[str],
    reference_outputs: list[np.ndarray],
    actual_outputs: tuple[object, ...],
) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}
    if len(reference_outputs) != len(actual_outputs):
        raise RuntimeError(f"{spec.name} produced {len(actual_outputs)} numeric outputs, expected {len(reference_outputs)}")

    for name, reference, actual in zip(output_names, reference_outputs, actual_outputs):
        reference_array = np.asarray(reference)
        actual_array = np.asarray(getattr(actual, "data", actual))
        if actual_array.shape != reference_array.shape:
            raise RuntimeError(f"{spec.name}.{name} numeric shape {actual_array.shape} != reference {reference_array.shape}")

        if np.issubdtype(reference_array.dtype, np.floating) or np.issubdtype(actual_array.dtype, np.floating):
            max_abs, max_rel = _numeric_diff(reference_array, actual_array)
            if not np.allclose(actual_array, reference_array, rtol=spec.numeric_rtol, atol=spec.numeric_atol, equal_nan=True):
                raise RuntimeError(
                    f"{spec.name}.{name} numeric mismatch: max_abs={max_abs:.3e}, "
                    f"max_rel={max_rel:.3e}, atol={spec.numeric_atol}, rtol={spec.numeric_rtol}"
                )
            checks[name] = {"shape": actual_array.shape, "max_abs": max_abs, "max_rel": max_rel}
        else:
            if not np.array_equal(actual_array, reference_array):
                raise RuntimeError(f"{spec.name}.{name} numeric mismatch for non-floating output")
            checks[name] = {"shape": actual_array.shape, "max_abs": 0.0, "max_rel": 0.0}
    return checks


def verify_model(path: Path, spec: ModelSpec, check_numeric: bool = True) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    output_names = [output.name for output in model.graph.output]
    ops = ONNXImport(str(path), strict=True)
    generic_nodes = [op for op in ops if isinstance(op, GenericNode)]
    if generic_nodes:
        names = ", ".join(f"{op.op_type}:{op.name}" for op in generic_nodes)
        raise RuntimeError(f"{spec.name} imported GenericNode fallback(s): {names}")

    initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(str(path))
    graph = Graph(ops=ops, input_name=initial_inputs, output_name=output_names, model_name=spec.name)
    feeds, runtime_inputs, placeholders = _prepare_inputs(initial_inputs, initial_tensors)
    outputs = _as_tuple(graph.forward_(*placeholders))

    if len(outputs) != len(output_names):
        raise RuntimeError(f"{spec.name} produced {len(outputs)} outputs, expected {len(output_names)}")
    for name, output in zip(output_names, outputs):
        expected_shape = spec.expected_outputs.get(name)
        if expected_shape is None:
            raise RuntimeError(f"{spec.name} has unexpected output {name}")
        if tuple(output.size) != expected_shape:
            raise RuntimeError(f"{spec.name}.{name} shape {tuple(output.size)} != {expected_shape}")

    op_counts: dict[str, int] = {}
    for op in ops:
        name = op.__class__.__name__
        op_counts[name] = op_counts.get(name, 0) + 1

    numeric_checks = {}
    if check_numeric:
        reference_outputs = ReferenceEvaluator(model).run(None, feeds)
        actual_outputs = _as_tuple(graph.forward(*runtime_inputs))
        numeric_checks = _compare_numeric_outputs(spec, output_names, reference_outputs, actual_outputs)

    return {"name": spec.name, "path": str(path), "outputs": output_names, "op_counts": op_counts, "numeric": numeric_checks}


def verify_model_suite(output_dir: Path, check_numeric: bool = True) -> list[dict[str, object]]:
    paths = generate_model_suite(output_dir)
    results = []
    for spec in MODEL_SPECS:
        result = verify_model(paths[spec.name], spec, check_numeric=check_numeric)
        results.append(result)
        op_summary = ", ".join(f"{name}={count}" for name, count in sorted(result["op_counts"].items()))
        print(f"PASS {spec.name}: {spec.description}")
        print(f"  model: {result['path']}")
        print(f"  ops: {op_summary}")
        if check_numeric:
            numeric_summary = ", ".join(
                f"{name}: abs={stats['max_abs']:.2e}, rel={stats['max_rel']:.2e}"
                for name, stats in result["numeric"].items()
            )
            print(f"  numeric: {numeric_summary}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and verify representative ONNX model smoke tests.")
    parser.add_argument("--output-dir", default=str(ROOT / "onnx_model" / "model_suite"), help="Directory for generated ONNX models.")
    parser.add_argument("--skip-numeric", action="store_true", help="Only verify import and shape inference; skip ONNX reference numeric comparison.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep generated ONNX model files for inspection.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(nn.TENSOR_OPS_LIB_PATH).exists():
        print(f"ERROR: C backend library not found: {nn.TENSOR_OPS_LIB_PATH}. Run `make` first.", file=sys.stderr)
        return 1
    output_dir = Path(args.output_dir)
    try:
        verify_model_suite(output_dir, check_numeric=not args.skip_numeric)
    finally:
        if not args.keep_artifacts and output_dir.exists():
            shutil.rmtree(output_dir)
            parent = output_dir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
    print("Representative model suite smoke gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
