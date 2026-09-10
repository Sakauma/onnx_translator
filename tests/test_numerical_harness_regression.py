import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.numerical import cli as numerical_cli
from tools.numerical import compare as numerical_compare
from tools.numerical import cuda as cuda_runner
from tools.numerical import runner as numerical_runner
from tools.numerical import runner_nps
from tools.numerical import runner_special_outputs


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def _write_executable(path, source):
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _isolated_temp_root(tmp_path, monkeypatch):
    temp_root = tmp_path / "invocations"
    temp_root.mkdir()
    monkeypatch.setattr(cuda_runner.tempfile, "tempdir", str(temp_root))
    return temp_root


@pytest.mark.parametrize("iterations", ["0", "-1"])
def test_numerical_cli_rejects_nonpositive_iterations_with_rc2(iterations):
    completed = subprocess.run(
        [sys.executable, "tools/cli.py", "numerical", "--iterations", iterations, "--skip-plots"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "iterations must be a positive integer" in completed.stderr
    assert "Pass (0/0)" not in completed.stdout


@pytest.mark.parametrize("iterations", [0, -1])
def test_direct_verify_op_rejects_nonpositive_iterations(iterations):
    with pytest.raises(ValueError, match="positive integer"):
        numerical_runner.verify_op(object, "add", [(1,)], ["float32"], "float32", iterations=iterations)


def test_unknown_numerical_operator_remains_rc2(monkeypatch):
    monkeypatch.setattr(numerical_cli.os.path, "exists", lambda _path: True)
    with pytest.raises(SystemExit) as caught:
        numerical_cli.main(["--op", "definitely_unknown", "--iterations", "1", "--skip-plots"])

    assert caught.value.code == 2


def test_verify_all_rejects_before_cleaning(tmp_path):
    copied_root = tmp_path / "repo"
    tools_dir = copied_root / "tools"
    cache_dir = copied_root / "cache"
    tools_dir.mkdir(parents=True)
    cache_dir.mkdir()
    shutil.copy2(ROOT / "tools" / "verify_all.py", tools_dir / "verify_all.py")
    sentinel = cache_dir / "sentinel.keep"
    sentinel.write_text("must survive argument rejection", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(tools_dir / "verify_all.py"), "--iterations", "0"],
        cwd=copied_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert sentinel.read_text(encoding="utf-8") == "must survive argument rejection"


@pytest.mark.parametrize(
    ("out_dtype", "nps_value", "wire_value"),
    [
        ("uint32", 16777216, 16777217),
        ("uint32", 16777217, 16777216),
        ("uint64", 2**53, 2**53 + 1),
        ("uint64", 2**64 - 1, 0),
    ],
)
def test_typed_unsigned_integer_wire_compares_exactly(out_dtype, nps_value, wire_value):
    dtype = np.dtype(out_dtype)
    ok, _mask, _nps, _wire, reason = numerical_compare.compare_integer_output(
        np.array([nps_value], dtype=dtype),
        np.array([wire_value], dtype=dtype),
        out_dtype,
    )

    assert not ok
    assert reason == "integer values differ"


@pytest.mark.parametrize(
    ("out_dtype", "value"),
    [("int32", 2**24 + 1), ("int64", 2**53 + 1), ("uint64", 2**64 - 1)],
)
def test_typed_integer_wire_accepts_exact_signed_and_unsigned_controls(out_dtype, value):
    dtype = np.dtype(out_dtype)
    ok, mask, _nps, _wire, reason = numerical_compare.compare_integer_output(
        np.array([value], dtype=dtype),
        np.array([value], dtype=dtype),
        out_dtype,
    )

    assert ok
    assert mask is None
    assert reason is None


@pytest.mark.parametrize(
    ("out_dtype", "nps", "wire", "expected_ok", "reason_fragment"),
    [
        ("uint64", np.array([7], dtype=np.uint64), np.array([7], dtype=np.int64), True, None),
        ("uint64", np.array([2**64 - 1], dtype=np.uint64), np.array([-1], dtype=np.int64), False, "out of range"),
        ("int64", np.array([2**63 - 1], dtype=np.int64), np.array([2**64 - 1], dtype=np.uint64), False, "out of range"),
    ],
)
def test_mixed_signed_integer_wires_are_range_checked_before_cast(
    out_dtype, nps, wire, expected_ok, reason_fragment
):
    ok, _mask, _nps, _wire, reason = numerical_compare.compare_integer_output(
        nps, wire, out_dtype
    )

    assert ok is expected_ok
    if reason_fragment is None:
        assert reason is None
    else:
        assert reason_fragment in reason


@pytest.mark.parametrize(
    ("out_dtype", "wire", "reason_fragment"),
    [
        ("uint32", np.array([16777216], dtype=np.float32), "exact precision"),
        ("uint32", np.array([16777217], dtype=np.float32), "exact precision"),
        ("uint64", np.array([2**53], dtype=np.float64), "exact precision"),
        ("uint64", np.array([float(2**64)], dtype=np.float64), "exact precision"),
        ("uint32", np.array([np.nan], dtype=np.float32), "non-finite"),
        ("uint32", np.array([np.inf], dtype=np.float32), "non-finite"),
        ("uint32", np.array([1.5], dtype=np.float32), "fractional"),
        ("uint8", np.array([256.0], dtype=np.float32), "out of range"),
        ("int8", np.array([-129.0], dtype=np.float32), "out of range"),
    ],
)
def test_floating_integer_wire_rejects_untrustworthy_values(out_dtype, wire, reason_fragment):
    nps = np.zeros(wire.shape, dtype=np.dtype(out_dtype))
    ok, _mask, _nps, _wire, reason = numerical_compare.compare_integer_output(nps, wire, out_dtype)

    assert not ok
    assert reason_fragment in reason


def test_finite_integer_difference_is_not_reported_as_nan_or_inf(monkeypatch, capsys):
    sample = np.array([7], dtype=np.int32)
    monkeypatch.setattr(numerical_runner, "prepare_input_samples", lambda *_args: [sample])
    monkeypatch.setattr(
        numerical_runner,
        "run_nps_forward",
        lambda *_args: SimpleNamespace(output=sample, topk_indices=None),
    )
    monkeypatch.setattr(numerical_runner, "build_cuda_params", lambda *_args: b"")
    monkeypatch.setattr(numerical_runner, "build_cuda_inputs", lambda *_args: [sample])
    monkeypatch.setattr(
        numerical_runner,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: np.array([8], dtype=np.int32),
    )

    _abs, _rel, ok = numerical_runner.verify_op(
        object, "probe", [(1,)], ["int32"], "int32", iterations=1
    )

    output = capsys.readouterr().out
    assert not ok
    assert "integer values differ" in output
    assert "NaN/Inf" not in output


def test_fractional_nps_integer_output_fails_through_verify_op(monkeypatch, capsys):
    sample = np.array([1], dtype=np.uint32)
    product_output = np.array([1.5], dtype=np.float32)
    monkeypatch.setattr(numerical_runner, "prepare_input_samples", lambda *_args: [sample])
    monkeypatch.setattr(
        numerical_runner,
        "run_nps_forward",
        lambda *_args: SimpleNamespace(output=product_output, topk_indices=None),
    )
    monkeypatch.setattr(numerical_runner, "build_cuda_params", lambda *_args: b"")
    monkeypatch.setattr(numerical_runner, "build_cuda_inputs", lambda *_args: [sample])
    monkeypatch.setattr(
        numerical_runner,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: np.array([1], dtype=np.uint32),
    )

    _abs, _rel, ok = numerical_runner.verify_op(
        object, "probe", [(1,)], ["uint32"], "uint32", iterations=1
    )

    output = capsys.readouterr().out
    assert not ok
    assert "NPS output dtype mismatch" in output
    assert "NaN/Inf" not in output


def test_reduce_output_normalization_preserves_wide_integer_dtype():
    value = np.array(2**63 + 1, dtype=np.uint64)
    normalized = runner_nps._normalize_reduce_output("reduce_max", value)

    assert normalized.dtype == np.uint64
    np.testing.assert_array_equal(normalized, np.array([2**63 + 1], dtype=np.uint64))


def _patch_verify_inputs(monkeypatch, nps_output, topk_indices=None):
    sample = np.array([16777217], dtype=np.uint32)
    monkeypatch.setattr(numerical_runner, "prepare_input_samples", lambda *_args: [sample])
    monkeypatch.setattr(
        numerical_runner,
        "run_nps_forward",
        lambda *_args: SimpleNamespace(output=nps_output, topk_indices=topk_indices),
    )
    monkeypatch.setattr(numerical_runner, "build_cuda_params", lambda *_args: b"")
    monkeypatch.setattr(numerical_runner, "build_cuda_inputs", lambda *_args: [sample])


def test_split_uint32_precision_boundary_fails_through_verify_op(monkeypatch, capsys):
    expected = np.array([16777217], dtype=np.uint32)
    _patch_verify_inputs(monkeypatch, [expected])
    monkeypatch.setattr(
        runner_special_outputs,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: np.array([16777216], dtype=np.float32),
    )

    _abs, _rel, ok = numerical_runner.verify_op(
        object, "split", [(1,)], ["uint32"], "uint32", iterations=1
    )

    assert not ok
    assert "exceeds exact precision" in capsys.readouterr().out


def test_unique_uint32_precision_boundary_fails_through_verify_op(monkeypatch, capsys):
    expected = np.array([16777217], dtype=np.uint32)
    indices = np.array([0], dtype=np.int64)
    inverse = np.array([0], dtype=np.int64)
    counts = np.array([1], dtype=np.int64)
    _patch_verify_inputs(monkeypatch, [expected, indices, inverse, counts])
    monkeypatch.setattr(
        runner_special_outputs,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: cuda_runner.CudaRunResult(
            output=np.array([16777216], dtype=np.float32),
            sidecars={
                "tmp_unique_indices.bin": indices,
                "tmp_unique_inverse.bin": inverse,
                "tmp_unique_counts.bin": counts,
            },
        ),
    )

    _abs, _rel, ok = numerical_runner.verify_op(
        object, "unique", [(1,)], ["uint32"], "uint32", iterations=1
    )

    assert not ok
    assert "exceeds exact precision" in capsys.readouterr().out


def test_topk_uint64_precision_boundary_fails_through_verify_op(monkeypatch, capsys):
    expected = np.array([2**53 + 1], dtype=np.uint64)
    indices = np.array([0], dtype=np.int64)
    _patch_verify_inputs(monkeypatch, expected, topk_indices=indices)
    monkeypatch.setattr(
        numerical_runner,
        "run_cuda_ground_truth",
        lambda *_args, **_kwargs: cuda_runner.CudaRunResult(
            output=np.array([2**53], dtype=np.float64),
            sidecars={"tmp_out_idx.bin": indices},
        ),
    )

    _abs, _rel, ok = numerical_runner.verify_op(
        object, "topk", [(1,)], ["uint32"], "uint64", iterations=1
    )

    assert not ok
    assert "exceeds exact precision" in capsys.readouterr().out


def test_runner_isolates_concurrent_calls_in_same_cwd(tmp_path, monkeypatch):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    _write_executable(
        verifier_dir / "verify_probe",
        f"#!{sys.executable}\n"
        "import pathlib, struct, sys, time\n"
        "value = pathlib.Path(sys.argv[2]).read_bytes()\n"
        "time.sleep(0.05)\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(value)\n",
    )
    temp_root = _isolated_temp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))

    def invoke(value):
        return cuda_runner.run_cuda_ground_truth(
            "probe", [np.array([value], dtype=np.float32)], target_shape=(1,)
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, [1003.0, 2003.0]))

    assert results == [1003.0, 2003.0]
    assert list(temp_root.iterdir()) == []


def test_runner_isolates_concurrent_multioutput_sidecars(tmp_path, monkeypatch):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    _write_executable(
        verifier_dir / "verify_probe",
        f"#!{sys.executable}\n"
        "import pathlib, sys, time\n"
        "value = pathlib.Path(sys.argv[2]).read_bytes()\n"
        "time.sleep(0.05)\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(value)\n"
        "pathlib.Path('tmp_probe_side.bin').write_bytes(value)\n",
    )
    temp_root = _isolated_temp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))
    spec = cuda_runner.CudaSidecarSpec("tmp_probe_side.bin", np.float32, (1,))

    def invoke(value):
        return cuda_runner.run_cuda_ground_truth(
            "probe",
            [np.array([value], dtype=np.float32)],
            target_shape=(1,),
            sidecars=[spec],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, [11.0, 29.0]))

    assert [result.output.item() for result in results] == [11.0, 29.0]
    assert [result.sidecars[spec.name].item() for result in results] == [11.0, 29.0]
    assert list(temp_root.iterdir()) == []


@pytest.mark.parametrize("mode", ["missing", "short", "extra"])
def test_malformed_sidecar_preserves_stderr_and_cleans(tmp_path, monkeypatch, mode):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    sidecar_statement = {
        "missing": "pass",
        "short": "pathlib.Path('tmp_side.bin').write_bytes(b'x')",
        "extra": "pathlib.Path('tmp_side.bin').write_bytes(b'12345')",
    }[mode]
    _write_executable(
        verifier_dir / "verify_probe",
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "print('sidecar diagnostic', file=sys.stderr)\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'1234')\n"
        f"{sidecar_statement}\n",
    )
    temp_root = _isolated_temp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))

    with pytest.raises(cuda_runner.CudaVerifierError) as caught:
        cuda_runner.run_cuda_ground_truth(
            "probe",
            [np.array([1], dtype=np.float32)],
            target_shape=(1,),
            sidecars=[cuda_runner.CudaSidecarSpec("tmp_side.bin", np.float32, (1,))],
        )

    assert caught.value.stderr == "sidecar diagnostic"
    assert "sidecar" in str(caught.value)
    assert list(temp_root.iterdir()) == []


def test_input_write_failure_cleans_invocation_directory(tmp_path, monkeypatch):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    _write_executable(verifier_dir / "verify_probe", "#!/bin/sh\nexit 0\n")
    temp_root = _isolated_temp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))

    def fail(*_args, **_kwargs):
        raise cuda_runner.CudaVerifierError("probe", "synthetic write failure")

    monkeypatch.setattr(cuda_runner, "_write_array_input", fail)
    with pytest.raises(cuda_runner.CudaVerifierError, match="synthetic write failure"):
        cuda_runner.run_cuda_ground_truth("probe", [np.array([1], dtype=np.float32)])

    assert list(temp_root.iterdir()) == []


def test_verifier_failure_preserves_stderr_and_cleans(tmp_path, monkeypatch):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    _write_executable(
        verifier_dir / "verify_probe",
        "#!/bin/sh\nprintf 'distinct verifier error\\n' >&2\nexit 9\n",
    )
    temp_root = _isolated_temp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))

    with pytest.raises(cuda_runner.CudaVerifierError) as caught:
        cuda_runner.run_cuda_ground_truth("probe", [np.array([1], dtype=np.float32)])

    assert caught.value.returncode == 9
    assert caught.value.stderr == "distinct verifier error"
    assert list(temp_root.iterdir()) == []


@pytest.mark.parametrize("payload", [b"", b"x", b"12345"])
def test_malformed_main_output_preserves_stderr_and_cleans(tmp_path, monkeypatch, payload):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    _write_executable(
        verifier_dir / "verify_probe",
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "print('main output diagnostic', file=sys.stderr)\n"
        f"pathlib.Path(sys.argv[-1]).write_bytes({payload!r})\n",
    )
    temp_root = _isolated_temp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))

    with pytest.raises(cuda_runner.CudaVerifierError) as caught:
        cuda_runner.run_cuda_ground_truth(
            "probe",
            [np.array([1], dtype=np.float32)],
            target_shape=(1,),
        )

    assert caught.value.stderr == "main output diagnostic"
    assert list(temp_root.iterdir()) == []


def test_real_gpu_add_and_unique_multioutput_cleanup(tmp_path, monkeypatch):
    add_exe = CACHE / "verify_add"
    unique_exe = CACHE / "verify_unique"
    if not add_exe.exists() or not unique_exe.exists():
        pytest.skip("required CUDA verifiers are not compiled")
    temp_root = _isolated_temp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(CACHE))

    try:
        add_inputs = [
            (np.array([1.25, -2.0], dtype=np.float32), np.array([2.75, 5.0], dtype=np.float32)),
            (np.array([1003.0, -7.0], dtype=np.float32), np.array([2003.0, 11.0], dtype=np.float32)),
        ]

        def run_add(pair):
            return cuda_runner.run_cuda_ground_truth("add", list(pair), target_shape=(2,))

        with ThreadPoolExecutor(max_workers=2) as pool:
            add_results = list(pool.map(run_add, add_inputs))
        source = np.array([2.0, -1.0, 2.0, 0.5, -1.0], dtype=np.float32)
        values, indices, inverse, counts = np.unique(
            source, return_index=True, return_inverse=True, return_counts=True
        )
        params = np.array([0, 1, source.size], dtype=np.int32).tobytes()
        unique = cuda_runner.run_cuda_ground_truth(
            "unique",
            [source],
            params_binary=params,
            output_dtype=np.float32,
            target_shape=values.shape,
            sidecars=[
                cuda_runner.CudaSidecarSpec("tmp_unique_indices.bin", np.int64, indices.shape),
                cuda_runner.CudaSidecarSpec("tmp_unique_inverse.bin", np.int64, inverse.shape),
                cuda_runner.CudaSidecarSpec("tmp_unique_counts.bin", np.int64, counts.shape),
            ],
        )
    except cuda_runner.CudaVerifierError as exc:
        if any(
            marker in exc.stderr
            for marker in ("no CUDA-capable device", "driver version is insufficient", "initialization error")
        ):
            pytest.skip(exc.stderr)
        raise

    np.testing.assert_array_equal(add_results[0], np.array([4.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(add_results[1], np.array([3006.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(unique.output, values)
    np.testing.assert_array_equal(unique.sidecars["tmp_unique_indices.bin"], indices)
    np.testing.assert_array_equal(unique.sidecars["tmp_unique_inverse.bin"], inverse)
    np.testing.assert_array_equal(unique.sidecars["tmp_unique_counts.bin"], counts)
    assert list(temp_root.iterdir()) == []
