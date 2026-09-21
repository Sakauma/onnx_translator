from types import SimpleNamespace

import numpy as np
import pytest

import nn
from tools.numerical import runner
from tools.numerical.runner_cuda_params import build_cuda_params


def _verify_ordinary_output(monkeypatch, *, out_dtype, nps_output, cuda_output):
    sample = np.array([1.0], dtype=np.float32)
    monkeypatch.setattr(runner, "prepare_input_samples", lambda *_args, **_kwargs: [sample])
    monkeypatch.setattr(
        runner,
        "run_nps_forward",
        lambda *_args, **_kwargs: SimpleNamespace(output=nps_output, topk_indices=None),
    )
    monkeypatch.setattr(runner, "build_cuda_params", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(runner, "build_cuda_inputs", lambda *_args, **_kwargs: [sample])
    monkeypatch.setattr(runner, "run_cuda_ground_truth", lambda *_args, **_kwargs: cuda_output)
    monkeypatch.setattr(runner, "resolve_output_shapes", lambda *_args, **_kwargs: ((1,),))
    return runner.verify_op(
        object, "probe", [(1,)], ["float32"], out_dtype, iterations=1
    )[2]


@pytest.mark.parametrize(
    ("out_dtype", "correct", "cuda_output"),
    [
        ("bool", np.array([False], dtype=np.bool_), np.array([0], dtype=np.uint8)),
        ("float16", np.array([1.0], dtype=np.float16), np.array([1.0], dtype=np.float32)),
        ("float32", np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
        ("float64", np.array([1.0], dtype=np.float64), np.array([1.0], dtype=np.float64)),
        ("bfloat16", np.array([0x3F80], dtype=np.uint16), np.array([1.0], dtype=np.float32)),
    ],
)
def test_ordinary_output_accepts_declared_storage_dtype(
    monkeypatch, out_dtype, correct, cuda_output
):
    assert correct.dtype == np.dtype(nn.DTYPE_TO_NUMPY[out_dtype])
    assert _verify_ordinary_output(
        monkeypatch, out_dtype=out_dtype, nps_output=correct, cuda_output=cuda_output
    )


@pytest.mark.parametrize(
    ("out_dtype", "wrong"),
    [
        ("bool", np.array([0.0], dtype=np.float32)),
        ("float16", np.array([1.0], dtype=np.float32)),
        ("float32", np.array([1.0], dtype=np.float64)),
        ("float64", np.array([1.0], dtype=np.float32)),
        ("bfloat16", np.array([1.0], dtype=np.float32)),
    ],
)
def test_ordinary_output_rejects_wrong_storage_dtype(monkeypatch, capsys, out_dtype, wrong):
    passed = _verify_ordinary_output(
        monkeypatch,
        out_dtype=out_dtype,
        nps_output=wrong,
        cuda_output=np.array([0.0], dtype=np.float32),
    )
    assert not passed
    assert "Output contract mismatch" in capsys.readouterr().out


def test_layer_normalization_cuda_params_include_stash_type():
    inputs = [
        np.zeros((2, 3), dtype=np.float32),
        np.ones((3,), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
    ]
    payload = build_cuda_params(
        "layer_normalization", inputs,
        {"axis": -1, "epsilon": 1e-5, "emit_stats": 1, "stash_type": 16},
        [(2, 3), (3,), (3,)], ["float32", "float32", "float32"],
        "float32", np.zeros((2, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.frombuffer(payload[:28], dtype=np.int32),
        np.array([2, 3, 1, 1, 1, 16, 1], dtype=np.int32),
    )
    assert np.frombuffer(payload[28:], dtype=np.float32).item() == pytest.approx(1e-5)


def test_layer_normalization_large_offset_plan_and_inputs():
    from tools.numerical.cli import build_default_plans
    from tools.numerical.runner_inputs import prepare_input_samples

    matching = [
        plan for plan in build_default_plans()
        if plan[1] == "layer_normalization"
        and len(plan) == 6
        and plan[5].get("input_values") == [100000000.0, 100000000.0, 100000008.0]
    ]
    assert len(matching) == 1
    _op_cls, op_name, shapes, dtypes, _out_dtype, init_args = matching[0]
    inputs = prepare_input_samples(op_name, shapes, dtypes, init_args)
    np.testing.assert_array_equal(
        inputs[0], np.array([[100000000.0, 100000000.0, 100000008.0]], dtype=np.float32)
    )


def test_layer_normalization_large_offset_plan_runs_real_nps_path():
    from nn import Tensor
    from nn.Operators import LayerNormalization
    from tools.numerical.cli import build_default_plans
    from tools.numerical.runner_inputs import prepare_input_samples
    from tools.numerical.runner_nps import run_nps_forward

    plan = next(
        plan for plan in build_default_plans()
        if plan[1] == "layer_normalization"
        and len(plan) == 6
        and "input_values" in plan[5]
    )
    _op_cls, op_name, shapes, dtypes, out_dtype, init_args = plan
    inputs_np = prepare_input_samples(op_name, shapes, dtypes, init_args)
    inputs = [
        Tensor(*value.shape, dtype=dtype, data=value)
        for value, dtype in zip(inputs_np, dtypes)
    ]
    result = run_nps_forward(
        LayerNormalization, op_name, inputs, init_args, out_dtype
    )

    assert result.output.shape == (1, 3)
    assert result.output.dtype == np.float32
    assert np.all(np.isfinite(result.output))


@pytest.mark.parametrize(
    ("dtype_name", "onnx_code"),
    [("float32", 1), ("float16", 10), ("float64", 11), ("bfloat16", 16)],
)
def test_layer_normalization_cuda_params_encode_input_dtype_independently_of_stash(
    dtype_name, onnx_code
):
    storage_dtype = nn.DTYPE_TO_NUMPY[dtype_name]
    inputs = [
        np.zeros((1, 3), dtype=storage_dtype),
        np.ones((3,), dtype=storage_dtype),
        np.zeros((3,), dtype=storage_dtype),
    ]
    payload = build_cuda_params(
        "layer_normalization", inputs,
        {"axis": -1, "epsilon": 1e-5, "stash_type": 1},
        [(1, 3), (3,), (3,)], [dtype_name, dtype_name, dtype_name],
        dtype_name, np.zeros((1, 3), dtype=storage_dtype),
    )
    fields = np.frombuffer(payload[:28], dtype=np.int32)
    assert fields[5] == 1
    assert fields[6] == onnx_code
