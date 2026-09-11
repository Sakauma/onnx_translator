"""Numerical-gate coverage for the ONNX ReferenceEvaluator DQL zero profile."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.numerical import cli, runner
from tools.numerical import runner_nps
from tools.numerical import runner_special_outputs as special
from tools.numerical.runner_inputs import prepare_input_samples


def _zero_reference_plans():
    return [
        plan
        for plan in cli.build_default_plans()
        if plan[1] == "dynamic_quantize_linear"
        and len(plan) == 6
        and plan[5].get("dql_zero_reference_profile")
    ]


def test_zero_reference_plans_generate_exact_zero_and_signed_zero_inputs():
    plans = _zero_reference_plans()
    assert len(plans) == 2

    generated = []
    for _op_cls, op_name, shapes, dtypes, _out_dtype, init_args in plans:
        values = prepare_input_samples(op_name, shapes, dtypes, init_args)[0]
        assert values.dtype == np.float32
        assert values.shape == shapes[0]
        assert values.size > 0
        assert np.all(np.isfinite(values))
        assert np.all(values == 0.0)
        generated.append(values)

    assert not np.any(np.signbit(generated[0]))
    assert np.any(np.signbit(generated[1]))
    assert np.any(~np.signbit(generated[1]))


@pytest.mark.parametrize("bad_values", [[0.0, 1.0], [0.0, np.nan]])
def test_zero_reference_plan_rejects_non_profile_input(bad_values):
    with pytest.raises(ValueError, match="finite, non-empty, all-zero"):
        prepare_input_samples(
            "dynamic_quantize_linear",
            [(2,)],
            ["float32"],
            {"input_values": bad_values, "dql_zero_reference_profile": True},
        )


def _verify_profile(
    monkeypatch,
    *,
    nps_scale,
    cuda_scale,
    nps_y=0,
    cuda_y=0,
    nps_zp=0,
    cuda_zp=0,
    profile_marker=True,
):
    sample = np.array([0.0, -0.0, 0.0, -0.0], dtype=np.float32)
    nps_output = [
        np.full(sample.shape, nps_y, dtype=np.uint8),
        np.array(nps_scale, dtype=np.float32),
        np.array(nps_zp, dtype=np.uint8),
    ]
    cuda_wire = np.concatenate(
        [
            np.full(sample.size, cuda_y, dtype=np.float32),
            np.array([cuda_scale, cuda_zp], dtype=np.float32),
        ]
    )
    monkeypatch.setattr(runner, "prepare_input_samples", lambda *_args, **_kwargs: [sample])
    monkeypatch.setattr(
        runner,
        "run_nps_forward",
        lambda *_args, **_kwargs: SimpleNamespace(output=nps_output, topk_indices=None),
    )
    monkeypatch.setattr(runner, "build_cuda_params", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(special, "run_cuda_ground_truth", lambda *_args, **_kwargs: cuda_wire)
    return runner.verify_op(
        object,
        "dynamic_quantize_linear",
        [sample.shape],
        ["float32"],
        "uint8",
        init_args={"dql_zero_reference_profile": True} if profile_marker else {},
        iterations=1,
    )[2]


def test_zero_reference_profile_accepts_bit_exact_three_outputs(monkeypatch):
    scale = special.DQL_ZERO_REFERENCE_SCALE
    assert _verify_profile(monkeypatch, nps_scale=scale, cuda_scale=scale) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"nps_scale": np.float32(1.0), "cuda_scale": np.float32(1.0)},
        {"nps_scale": np.float32(0.0), "cuda_scale": np.float32(0.0)},
        {
            "nps_scale": np.nextafter(special.DQL_ZERO_REFERENCE_SCALE, np.float32(np.inf)),
            "cuda_scale": np.nextafter(special.DQL_ZERO_REFERENCE_SCALE, np.float32(np.inf)),
        },
        {"nps_y": 1, "cuda_y": 1},
        {"nps_zp": 1, "cuda_zp": 1},
    ],
    ids=["scale-one", "scale-zero", "scale-one-ulp", "y-one", "zero-point-one"],
)
def test_zero_reference_profile_rejects_agreeing_non_reference_outputs(monkeypatch, overrides):
    scale = special.DQL_ZERO_REFERENCE_SCALE
    output_overrides = {
        key: value for key, value in overrides.items() if key not in {"nps_scale", "cuda_scale"}
    }
    assert _verify_profile(
        monkeypatch,
        nps_scale=overrides.get("nps_scale", scale),
        cuda_scale=overrides.get("cuda_scale", scale),
        **output_overrides,
    ) is False


@pytest.mark.parametrize(
    "bad_scale",
    [
        np.float32(1.0),
        np.float32(0.0),
        np.nextafter(special.DQL_ZERO_REFERENCE_SCALE, np.float32(np.inf)),
    ],
    ids=["scale-one", "scale-zero", "scale-one-ulp"],
)
def test_unmarked_zero_input_still_rejects_agreeing_bad_scale(monkeypatch, bad_scale):
    assert _verify_profile(
        monkeypatch,
        nps_scale=bad_scale,
        cuda_scale=bad_scale,
        profile_marker=False,
    ) is False


def test_zero_reference_marker_is_harness_only_constructor_metadata():
    init_args = {
        "input_values": [0.0, -0.0],
        "dql_zero_reference_profile": True,
    }
    op_init_args, _controls = runner_nps._operator_init_args(
        "dynamic_quantize_linear", init_args, "uint8"
    )

    assert "dql_zero_reference_profile" not in op_init_args
    assert init_args["dql_zero_reference_profile"] is True
