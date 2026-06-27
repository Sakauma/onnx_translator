# /**
#   ******************************************************************************
#   * @file        test_model_suite.py
#   * @author      Egor Izmaylov
#   * @brief       Verifies representative ONNX model suite generation.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import onnx
import numpy as np
import pytest

from tools.model_suite import MODEL_SPECS, _compare_numeric_outputs, _input_data, generate_model_suite


def _output_shapes(model):
    shapes = {}
    for output in model.graph.output:
        dims = tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
        shapes[output.name] = dims
    return shapes


def test_model_suite_generates_valid_representative_models(tmp_path):
    paths = generate_model_suite(tmp_path)

    assert set(paths) == {spec.name for spec in MODEL_SPECS}
    for spec in MODEL_SPECS:
        model = onnx.load(paths[spec.name], load_external_data=False)
        onnx.checker.check_model(model)
        assert _output_shapes(model) == spec.expected_outputs


def test_model_suite_covers_release_representative_operator_families(tmp_path):
    paths = generate_model_suite(tmp_path)
    ops_by_model = {
        name: {node.op_type for node in onnx.load(path, load_external_data=False).graph.node}
        for name, path in paths.items()
    }

    assert {"Conv", "BatchNormalization", "GlobalAveragePool", "Gemm", "Softmax"} <= ops_by_model["vision_cnn"]
    assert {"MatMul", "Transpose", "Softmax", "Add", "LayerNormalization"} <= ops_by_model["transformer_block"]
    assert {"Gather", "ReduceMean", "Concat", "Gemm", "Sigmoid"} <= ops_by_model["embedding_mlp"]


def test_model_suite_inputs_are_deterministic_and_dtype_correct():
    first = _input_data((2, 3), "int64")
    second = _input_data((2, 3), "int64")
    floats = _input_data((2, 2), "float32")

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64
    assert floats.dtype == np.float32


def test_numeric_comparison_reports_and_rejects_reference_mismatches():
    spec = MODEL_SPECS[0]
    actual = type("TensorLike", (), {"data": np.array([1.0, 1.00001], dtype=np.float32)})()

    checks = _compare_numeric_outputs(spec, ["out"], [np.array([1.0, 1.0], dtype=np.float32)], (actual,))
    assert checks["out"]["max_abs"] > 0.0

    bad = type("TensorLike", (), {"data": np.array([1.0, 2.0], dtype=np.float32)})()
    with pytest.raises(RuntimeError, match="numeric mismatch"):
        _compare_numeric_outputs(spec, ["out"], [np.array([1.0, 1.0], dtype=np.float32)], (bad,))
