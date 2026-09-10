# /**
#   ******************************************************************************
#   * @file        test_numerical_runner_architecture.py
#   * @author      Egor Izmaylov
#   * @brief       验证数值调度器拆分后的算子族配置和 CUDA 输入策略。
#   * @details     2026.07.15  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np
import pytest

from tools.numerical import cli as numerical_cli
from tools.numerical.runner_config import resolve_verification_config
from tools.numerical.runner_cuda_inputs import build_cuda_inputs, resolve_cuda_output_dtype
from tools.numerical.runner_inputs import prepare_input_samples


def test_all_default_slice_plans_prepare_inputs():
    slice_plans = [plan for plan in numerical_cli.build_default_plans() if plan[1] == "slice"]

    assert len(slice_plans) == 5
    for _op_cls, op_name, shapes, dtypes, _out_dtype, init_args in slice_plans:
        inputs = prepare_input_samples(op_name, shapes, dtypes, init_args)
        assert len(inputs) == 5
        assert inputs[1].dtype == np.int64
        assert inputs[2].dtype == np.int64
        assert inputs[3].dtype == np.int64
        assert inputs[4].dtype == np.int64


def test_cli_aggregates_plan_exception_and_runs_remaining_plan(monkeypatch, capsys):
    plans = [
        (object, "broken", [(1,)], ["float32"], "float32"),
        (object, "healthy", [(1,)], ["float32"], "float32"),
    ]
    calls = []

    def fake_verify_op(_cls, name, *_args, **_kwargs):
        calls.append(name)
        if name == "broken":
            raise RuntimeError("preparation exploded")
        return [], [], True

    monkeypatch.setattr(numerical_cli, "build_default_plans", lambda: plans)
    monkeypatch.setattr(numerical_cli, "verify_op", fake_verify_op)
    monkeypatch.setattr(numerical_cli.os.path, "exists", lambda _path: True)

    with pytest.raises(SystemExit) as exc_info:
        numerical_cli.main(["--iterations", "1", "--skip-plots"])

    assert exc_info.value.code == 1
    assert calls == ["broken", "healthy"]
    output = capsys.readouterr().out
    assert "preparation exploded" in output
    assert "numerical verification failed for: ['broken']" in output


@pytest.mark.parametrize(
    "op_name,out_dtype,expected",
    [
        ("add", "float32", (1e-4, 1e-4)),
        ("cos", "float32", (0.02, 1e-4)),
        ("einsum", "float32", (1e-2, 1e-3)),
        ("add", "float16", (0.01, 0.01)),
        ("add", "bfloat16", (0.1, 0.02)),
        ("add", "float8_e4m3", (0.1, 0.1)),
        ("gather", "int64", (0.0, 0.0)),
    ],
)
def test_verification_tolerances_are_resolved_from_config(op_name, out_dtype, expected):
    config = resolve_verification_config(op_name, out_dtype)

    assert (config.atol, config.rtol) == expected


def test_operator_family_config_controls_cuda_conversion():
    conv_config = resolve_verification_config("conv2d", "float32")
    gather_config = resolve_verification_config("gather", "int64")
    add_config = resolve_verification_config("add", "float32")

    assert conv_config.complex_kernel
    assert conv_config.double_kernel
    assert gather_config.int64_passthrough
    assert not gather_config.broadcast_inputs
    assert add_config.broadcast_inputs

    broadcast = build_cuda_inputs(
        "add",
        [np.asarray([1.0, 2.0], dtype=np.float32)],
        ["float32"],
        {},
        (2, 2),
        add_config,
    )[0]
    passthrough = build_cuda_inputs(
        "gather",
        [np.asarray([1, 2], dtype=np.int64)],
        ["int64"],
        {},
        (2,),
        gather_config,
    )[0]

    assert broadcast.shape == (2, 2)
    assert broadcast.dtype == np.float32
    assert passthrough.dtype == np.int64
    assert resolve_cuda_output_dtype("gather", "int64", gather_config) == np.int64
