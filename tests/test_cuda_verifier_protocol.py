import os
import stat
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.numerical import cuda as cuda_runner


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def _executable(name):
    path = CACHE / f"verify_{name}"
    if not path.exists():
        pytest.skip(f"CUDA verifier is not compiled: {path}")
    return path


def _write_executable(path, source):
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_runner_preserves_exit_code_and_distinctive_stderr(tmp_path, monkeypatch):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    _write_executable(
        verifier_dir / "verify_probe",
        "#!/bin/sh\nprintf 'distinctive verifier failure\\n' >&2\nexit 7\n",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))

    with pytest.raises(cuda_runner.CudaVerifierError) as caught:
        cuda_runner.run_cuda_ground_truth("probe", [np.array([1], dtype=np.float32)])

    assert caught.value.returncode == 7
    assert caught.value.stderr == "distinctive verifier failure"
    assert "exit code 7" in str(caught.value)
    assert "distinctive verifier failure" in str(caught.value)


@pytest.mark.parametrize("payload", [b"", b"\x00" * 3, b"\x00" * 5])
def test_runner_rejects_missing_truncated_and_extra_output(tmp_path, monkeypatch, payload):
    verifier_dir = tmp_path / "verifiers"
    verifier_dir.mkdir()
    payload_literal = repr(payload)
    _write_executable(
        verifier_dir / "verify_probe",
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path(__import__('sys').argv[-1]).write_bytes({payload_literal})\n",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(verifier_dir))

    with pytest.raises(cuda_runner.CudaVerifierError, match="invalid output size|did not create output"):
        cuda_runner.run_cuda_ground_truth(
            "probe", [np.array([1], dtype=np.float32)], target_shape=(1,)
        )


def test_add_without_visible_gpu_fails_with_diagnostic(tmp_path):
    executable = _executable("add")
    input_a = tmp_path / "a.bin"
    input_b = tmp_path / "b.bin"
    output = tmp_path / "out.bin"
    np.array([1.0], dtype=np.float32).tofile(input_a)
    np.array([2.0], dtype=np.float32).tofile(input_b)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    completed = subprocess.run(
        [str(executable), "1", str(input_a), str(input_b), str(output)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stderr.strip()
    assert not output.exists() or output.stat().st_size != np.dtype(np.float32).itemsize


def test_dynamic_quantize_gpu_exact_ties_to_even_fixture(tmp_path, monkeypatch):
    _executable("dynamic_quantize_linear")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(CACHE))
    source = np.array([-253.0, 257.0, 0.0], dtype=np.float32)

    try:
        packed = cuda_runner.run_cuda_ground_truth(
            "dynamic_quantize_linear",
            [source],
            output_dtype=np.float32,
            target_shape=(5,),
        )
    except cuda_runner.CudaVerifierError as exc:
        if exc.stderr and (
            "no CUDA-capable device" in exc.stderr
            or "CUDA driver version is insufficient" in exc.stderr
            or "initialization error" in exc.stderr
        ):
            pytest.skip(exc.stderr)
        raise

    y = packed[:3].astype(np.uint8)
    scale = np.asarray(packed[3], dtype=np.float32)
    zero_point = np.asarray(packed[4], dtype=np.uint8)
    assert y.dtype == np.uint8
    assert y.shape == (3,)
    np.testing.assert_array_equal(y, np.array([0, 254, 126], dtype=np.uint8))
    assert scale.dtype == np.float32
    assert scale.item() == np.float32(2.0)
    assert zero_point.dtype == np.uint8
    assert zero_point.item() == 126


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (np.array([np.inf], dtype=np.float32), np.inf),
        (np.array([np.inf, -np.inf], dtype=np.float32), np.inf),
        (np.array([-np.inf, -np.inf], dtype=np.float32), -np.inf),
        (np.array([np.nan, np.inf], dtype=np.float32), np.nan),
        (np.array([1000.0, 1000.0], dtype=np.float32), np.float32(1000.0 + np.log(2.0))),
    ],
)
def test_reduce_log_sum_exp_gpu_nonfinite_and_stable_semantics(
    tmp_path, monkeypatch, source, expected
):
    _executable("reduce_log_sum_exp")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(CACHE))

    try:
        result = cuda_runner.run_cuda_ground_truth(
            "reduce_log_sum_exp",
            [source],
            params_binary=struct.pack("q", source.size),
            output_dtype=np.float32,
            target_shape=(1,),
        )[0]
    except cuda_runner.CudaVerifierError as exc:
        if exc.stderr and (
            "no CUDA-capable device" in exc.stderr
            or "CUDA driver version is insufficient" in exc.stderr
            or "initialization error" in exc.stderr
        ):
            pytest.skip(exc.stderr)
        raise

    if np.isnan(expected):
        assert np.isnan(result)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def _run_unique(tmp_path, source, sorted_output=1):
    executable = _executable("unique")
    input_path = tmp_path / "input.bin"
    params_path = tmp_path / "params.bin"
    output_path = tmp_path / "values.bin"
    np.ascontiguousarray(source).tofile(input_path)
    type_code = 1 if source.dtype == np.int64 else 0
    params_path.write_bytes(struct.pack("iii", type_code, sorted_output, source.size))
    completed = subprocess.run(
        [str(executable), str(source.size), str(input_path), str(params_path), str(output_path)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed, output_path


def test_unique_gpu_writes_four_exact_outputs(tmp_path):
    source = np.array([2.0, -1.0, 2.0, 0.5, -1.0, 3.0, 0.5], dtype=np.float32)
    expected_values, expected_indices, expected_inverse, expected_counts = np.unique(
        source, return_index=True, return_inverse=True, return_counts=True
    )
    completed, output_path = _run_unique(tmp_path, source)
    if completed.returncode != 0 and (
        "no CUDA-capable device" in completed.stderr
        or "CUDA driver version is insufficient" in completed.stderr
        or "initialization error" in completed.stderr
    ):
        pytest.skip(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(np.fromfile(output_path, dtype=np.float32), expected_values)
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "tmp_unique_indices.bin", dtype=np.int64), expected_indices
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "tmp_unique_inverse.bin", dtype=np.int64), expected_inverse
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "tmp_unique_counts.bin", dtype=np.int64), expected_counts
    )


def test_unique_sidecar_write_failure_is_nonzero_and_names_path(tmp_path):
    blocked = tmp_path / "tmp_unique_counts.bin"
    blocked.mkdir()
    completed, _ = _run_unique(tmp_path, np.array([3, 1, 3, 2], dtype=np.int64))
    if completed.returncode != 0 and (
        "no CUDA-capable device" in completed.stderr
        or "CUDA driver version is insufficient" in completed.stderr
        or "initialization error" in completed.stderr
    ):
        pytest.skip(completed.stderr)
    assert completed.returncode != 0
    assert "tmp_unique_counts.bin" in completed.stderr
