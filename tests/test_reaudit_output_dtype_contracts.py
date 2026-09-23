from types import SimpleNamespace

import numpy as np

from nn import Tensor
from nn.Operators import Split, Unique
from tools.numerical import runner, runner_special_outputs
from tools.numerical.cli import build_default_plans
from tools.numerical.runner_inputs import prepare_input_samples
from tools.numerical.runner_nps import run_nps_forward
from tools.numerical.runner_shapes import (
    resolve_output_dtypes,
    resolve_output_shapes,
    validate_nps_output_shapes,
)


def test_multidimensional_unique_inverse_contract_matches_real_forward():
    data = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]], dtype=np.float32)
    expected_shapes = resolve_output_shapes("unique", [data], {})
    expected_dtypes = resolve_output_dtypes("unique", "float32", {}, len(expected_shapes))
    result = run_nps_forward(
        Unique,
        "unique",
        [Tensor(*data.shape, dtype="float32", data=data)],
        {},
        "float32",
    )

    assert expected_shapes == ((4,), (4,), (6,), (4,))
    assert expected_dtypes == ("float32", "int64", "int64", "int64")
    validate_nps_output_shapes("unique", result, expected_shapes, expected_dtypes)


def test_output_dtype_resolver_keeps_mixed_public_outputs():
    cases = [
        ("topk", "float16", {}, 2, ("float16", "int64")),
        ("dropout", "bfloat16", {}, 2, ("bfloat16", "bool")),
        ("dynamic_quantize_linear", "uint8", {}, 3, ("uint8", "float32", "uint8")),
        ("unique", "float32", {}, 4, ("float32", "int64", "int64", "int64")),
        ("layer_normalization", "float16", {"emit_stats": 1, "stash_type": 16}, 3,
         ("float16", "bfloat16", "bfloat16")),
        ("batch_normalization", "float16", {"training_mode": 1}, 3,
         ("float16", "float16", "float16")),
        ("softmax_cross_entropy_loss", "float16", {"emit_log_prob": 1}, 2,
         ("float16", "float16")),
        ("split", "float32", {"num_outputs": 3}, 3,
         ("float32", "float32", "float32")),
        ("lstm", "bfloat16", {}, 3, ("bfloat16", "bfloat16", "bfloat16")),
    ]
    for op_name, out_dtype, init_args, count, expected in cases:
        assert resolve_output_dtypes(op_name, out_dtype, init_args, count) == expected


def test_split_second_output_wrong_dtype_fails_before_cuda(monkeypatch, capsys):
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    split = np.array([1, 2], dtype=np.int64)
    outputs = [data[:, :1].copy(), data[:, 1:].astype(np.float64)]
    monkeypatch.setattr(runner, "prepare_input_samples", lambda *_args: [data, split])
    monkeypatch.setattr(
        runner,
        "run_nps_forward",
        lambda *_args: SimpleNamespace(output=outputs, topk_indices=None),
    )
    monkeypatch.setattr(
        runner_special_outputs,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wrong NPS dtype reached CUDA")
        ),
    )

    _abs, _rel, passed = runner.verify_op(
        Split,
        "split",
        [data.shape, split.shape],
        ["float32", "int64"],
        "float32",
        {"axis": 1, "num_outputs": 2},
        iterations=1,
    )

    assert not passed
    output = capsys.readouterr().out
    assert "output 1" in output
    assert "dtype mismatch" in output


def test_every_default_plan_real_nps_output_matches_independent_contract():
    plans = build_default_plans()
    assert len(plans) == 728
    for index, plan in enumerate(plans):
        op_cls, op_name, shapes, dtypes, out_dtype = plan[:5]
        init_args = plan[5] if len(plan) == 6 else {}
        inputs = prepare_input_samples(op_name, shapes, dtypes, init_args)
        expected_shapes = resolve_output_shapes(op_name, inputs, init_args)
        expected_dtypes = resolve_output_dtypes(
            op_name, out_dtype, init_args, len(expected_shapes)
        )
        tensors = [
            None if value is None else Tensor(*value.shape, dtype=dtype, data=value)
            for value, dtype in zip(inputs, dtypes)
        ]
        result = run_nps_forward(op_cls, op_name, tensors, init_args, out_dtype)
        try:
            validate_nps_output_shapes(
                op_name, result, expected_shapes, expected_dtypes
            )
        except Exception as exc:
            raise AssertionError(f"default plan {index} ({op_name}) failed: {exc}") from exc
