"""Regression coverage for REAUD-007 special-output contract validation."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.numerical import cli as numerical_cli
from tools.numerical import runner
from tools.numerical import runner_special_outputs as special
from tools.numerical.cuda import CudaRunResult


SAMPLE = np.array([0.25], dtype=np.float32)


def _install_common(monkeypatch, *, inputs, nps_output, topk_indices=None):
    monkeypatch.setattr(runner, "prepare_input_samples", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(
        runner,
        "run_nps_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            output=nps_output, topk_indices=topk_indices
        ),
    )
    monkeypatch.setattr(runner, "build_cuda_params", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(runner, "build_cuda_inputs", lambda *_args, **_kwargs: inputs)


def _verify_dql(monkeypatch, nps_output, cuda_wire):
    _install_common(monkeypatch, inputs=[SAMPLE], nps_output=nps_output)
    monkeypatch.setattr(
        special,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: np.asarray(cuda_wire, dtype=np.float32),
    )
    return runner.verify_op(
        object, "dynamic_quantize_linear", [(1,)], ["float32"], "uint8", iterations=1
    )[2]


VALID_DQL_NPS = [
    np.array([1], dtype=np.uint8),
    np.array(0.25, dtype=np.float32),
    np.array(7, dtype=np.uint8),
]


@pytest.mark.parametrize(
    ("nps_output", "cuda_wire"),
    [
        (
            [
                np.array([1.75], dtype=np.float32),
                np.array([0.25], dtype=np.float32),
                np.array([7.75], dtype=np.float32),
            ],
            [1.0, 0.25, 7.0],
        ),
        (VALID_DQL_NPS, [1.4, 0.25, 7.0]),
        (VALID_DQL_NPS, [300.0, 0.25, 7.0]),
        (VALID_DQL_NPS, [np.nan, 0.25, 7.0]),
        (VALID_DQL_NPS, [1.0, 0.25, 7.4]),
    ],
    ids=[
        "invalid-nps-dtype-and-scalar-shape",
        "fractional-cuda-y",
        "out-of-range-cuda-y",
        "nonfinite-cuda-y",
        "fractional-cuda-zero-point",
    ],
)
def test_dql_fault_injections_fail_verify_op(monkeypatch, nps_output, cuda_wire):
    assert _verify_dql(monkeypatch, nps_output, cuda_wire) is False


def test_topk_fractional_nps_indices_fail_verify_op(monkeypatch):
    _install_common(
        monkeypatch,
        inputs=[SAMPLE, np.array([1], dtype=np.int64)],
        nps_output=np.array([5.0], dtype=np.float32),
        topk_indices=np.array([0.75], dtype=np.float32),
    )
    monkeypatch.setattr(
        runner,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: CudaRunResult(
            output=np.array([5.0], dtype=np.float32),
            sidecars={"tmp_out_idx.bin": np.array([0], dtype=np.int64)},
        ),
    )
    assert runner.verify_op(
        object, "topk", [(1,), (1,)], ["float32", "int64"], "float32", iterations=1
    )[2] is False


def test_unique_fractional_nps_aux_outputs_fail_verify_op(monkeypatch):
    _install_common(
        monkeypatch,
        inputs=[np.array([3.0], dtype=np.float32)],
        nps_output=[
            np.array([3.0], dtype=np.float32),
            np.array([0.75], dtype=np.float32),
            np.array([0.75], dtype=np.float32),
            np.array([1.75], dtype=np.float32),
        ],
    )
    monkeypatch.setattr(
        special,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: CudaRunResult(
            output=np.array([3.0], dtype=np.float32),
            sidecars={
                "tmp_unique_indices.bin": np.array([0], dtype=np.int64),
                "tmp_unique_inverse.bin": np.array([0], dtype=np.int64),
                "tmp_unique_counts.bin": np.array([1], dtype=np.int64),
            },
        ),
    )
    assert runner.verify_op(
        object, "unique", [(1,)], ["float32"], "float32", iterations=1
    )[2] is False


def test_dropout_non_bool_nps_mask_fails_verify_op(monkeypatch):
    _install_common(
        monkeypatch,
        inputs=[SAMPLE, np.array([0.5], dtype=np.float32), np.array([True])],
        nps_output=[np.array([0.5], dtype=np.float32), np.array([2], dtype=np.uint8)],
    )
    monkeypatch.setattr(
        special,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: CudaRunResult(
            output=np.array([0.5], dtype=np.float32),
            sidecars={"tmp_dropout_mask.bin": np.array([1], dtype=np.uint8)},
        ),
    )
    assert runner.verify_op(
        object,
        "dropout",
        [(1,), (1,), (1,)],
        ["float32", "float32", "bool"],
        "float32",
        iterations=1,
    )[2] is False


def test_valid_dql_contract_still_passes(monkeypatch):
    assert _verify_dql(monkeypatch, VALID_DQL_NPS, [1.0, 0.25, 7.0]) is True


@pytest.mark.parametrize("op_name", ["topk", "unique", "dropout"])
def test_valid_typed_special_output_contracts_still_pass(monkeypatch, op_name):
    if op_name == "topk":
        _install_common(
            monkeypatch,
            inputs=[SAMPLE, np.array([1], dtype=np.int64)],
            nps_output=np.array([5.0], dtype=np.float32),
            topk_indices=np.array([0], dtype=np.int64),
        )
        monkeypatch.setattr(
            runner,
            "run_cuda_ground_truth",
            lambda *_args, **_kwargs: CudaRunResult(
                output=np.array([5.0], dtype=np.float32),
                sidecars={"tmp_out_idx.bin": np.array([0], dtype=np.int64)},
            ),
        )
        args = (object, "topk", [(1,), (1,)], ["float32", "int64"], "float32")
    elif op_name == "unique":
        outputs = [
            np.array([3.0], dtype=np.float32),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ]
        _install_common(monkeypatch, inputs=[np.array([3.0], dtype=np.float32)], nps_output=outputs)
        monkeypatch.setattr(
            special,
            "run_cuda_ground_truth",
            lambda *_args, **_kwargs: CudaRunResult(
                output=outputs[0],
                sidecars={
                    "tmp_unique_indices.bin": outputs[1],
                    "tmp_unique_inverse.bin": outputs[2],
                    "tmp_unique_counts.bin": outputs[3],
                },
            ),
        )
        args = (object, "unique", [(1,)], ["float32"], "float32")
    else:
        outputs = [np.array([0.5], dtype=np.float32), np.array([True], dtype=np.bool_)]
        _install_common(
            monkeypatch,
            inputs=[SAMPLE, np.array([0.5], dtype=np.float32), np.array([True])],
            nps_output=outputs,
        )
        monkeypatch.setattr(
            special,
            "run_cuda_ground_truth",
            lambda *_args, **_kwargs: CudaRunResult(
                output=outputs[0],
                sidecars={"tmp_dropout_mask.bin": np.array([1], dtype=np.uint8)},
            ),
        )
        args = (
            object,
            "dropout",
            [(1,), (1,), (1,)],
            ["float32", "float32", "bool"],
            "float32",
        )
    assert runner.verify_op(*args, iterations=1)[2] is True


def test_cli_exits_nonzero_when_contract_failure_reaches_plan_result(monkeypatch):
    plan = (object, "topk", [(1,), (1,)], ["float32", "int64"], "float32")
    monkeypatch.setattr(numerical_cli.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(numerical_cli, "build_default_plans", lambda: [plan])
    monkeypatch.setattr(
        numerical_cli, "verify_op", lambda *_args, **_kwargs: ([], [], False)
    )
    with pytest.raises(SystemExit) as caught:
        numerical_cli.main(["--iterations", "1", "--skip-plots"])
    assert caught.value.code == 1
