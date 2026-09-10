# /**
#   ******************************************************************************
#   * @file        cuda.py
#   * @author      Egor Izmaylov
#   * @brief       负责调用 CUDA verifier 可执行文件并读取二进制输出。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from dataclasses import dataclass
import os
import subprocess
import tempfile

import numpy as np


CUDA_VERIFY_DIR = os.environ.get("CUDA_VERIFY_DIR", "cache")


@dataclass(frozen=True)
class CudaSidecarSpec:
    """Describe one fixed-name file emitted beside a verifier's main output."""

    name: str
    dtype: object
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CudaRunResult:
    """In-memory result for a verifier invocation that requested sidecars."""

    output: np.ndarray
    sidecars: dict[str, np.ndarray]


class CudaVerifierError(RuntimeError):
    def __init__(self, op_name, message, returncode=None, stderr=""):
        self.op_name = op_name
        self.returncode = returncode
        self.stderr = stderr
        details = f"CUDA verifier failed [{op_name}]"
        if returncode is not None:
            details += f" with exit code {returncode}"
        if message:
            details += f": {message}"
        if stderr:
            details += f"\nstderr: {stderr}"
        super().__init__(details)


def _write_array_input(op_name, path, array):
    try:
        np.ascontiguousarray(array).tofile(path)
    except OSError as exc:
        raise CudaVerifierError(op_name, f"failed to write input {path}: {exc}") from exc


def _write_params_input(op_name, path, params_binary):
    try:
        with open(path, "wb") as params_file:
            params_file.write(params_binary)
    except OSError as exc:
        raise CudaVerifierError(op_name, f"failed to write params {path}: {exc}") from exc


def _read_array_output(op_name, path, dtype, shape, label, stderr=""):
    expected_count = int(np.prod(shape))
    expected_bytes = expected_count * np.dtype(dtype).itemsize
    if not os.path.exists(path):
        raise CudaVerifierError(
            op_name,
            f"verifier did not create {label}: {path}",
            stderr=stderr,
        )
    try:
        actual_bytes = os.path.getsize(path)
        if actual_bytes != expected_bytes:
            raise CudaVerifierError(
                op_name,
                f"invalid {label} size: expected {expected_bytes} bytes, got {actual_bytes}",
                stderr=stderr,
            )
        value = np.fromfile(path, dtype=dtype)
        if value.size != expected_count:
            raise CudaVerifierError(
                op_name,
                f"invalid {label} element count: expected {expected_count}, got {value.size}",
                stderr=stderr,
            )
        return value.reshape(shape).copy()
    except CudaVerifierError:
        raise
    except (OSError, ValueError) as exc:
        raise CudaVerifierError(op_name, f"failed to read {label}: {exc}", stderr=stderr) from exc


def run_cuda_ground_truth(
    op_name,
    inputs_f32,
    params_binary=None,
    output_dtype=np.float32,
    target_shape=None,
    sidecars=None,
):
    """Run one verifier in an isolated directory and return fully parsed outputs.

    The verifier executable and every command-line file path are absolute. Fixed-name
    sidecars are resolved relative to the invocation directory, parsed into memory, and
    validated before that directory is removed. Calls without sidecars retain the legacy
    ndarray return type; calls with sidecars return ``CudaRunResult``.
    """
    exe = os.path.abspath(os.path.join(CUDA_VERIFY_DIR, f"verify_{op_name}"))
    if not os.path.isfile(exe):
        raise CudaVerifierError(op_name, f"missing executable: {exe}")

    cuda_inputs = list(inputs_f32)
    sidecar_specs = tuple(sidecars or ())
    for spec in sidecar_specs:
        if os.path.basename(spec.name) != spec.name:
            raise CudaVerifierError(op_name, f"sidecar name must be a basename: {spec.name}")

    try:
        with tempfile.TemporaryDirectory(prefix="onnx-translator-cuda-") as artifact_dir:
            files = []
            for i, array in enumerate(cuda_inputs):
                if array is None:
                    files.append("null")
                    continue
                input_path = os.path.abspath(os.path.join(artifact_dir, f"tmp_in_{i}.bin"))
                _write_array_input(op_name, input_path, array)
                files.append(input_path)

            if params_binary is not None:
                params_path = os.path.abspath(os.path.join(artifact_dir, "tmp_params.bin"))
                _write_params_input(op_name, params_path, params_binary)
                files.append(params_path)

            output_path = os.path.abspath(os.path.join(artifact_dir, "tmp_out.bin"))
            output_count = (
                int(np.prod(target_shape))
                if target_shape is not None
                else int(cuda_inputs[0].size)
            )
            if op_name == "resize":
                params_path = files[-1] if params_binary is not None else None
                if params_path is None:
                    raise CudaVerifierError(op_name, "resize requires params_binary")
                args = [exe, str(output_count), files[0], params_path, output_path]
            else:
                args = [exe, str(output_count), *files, output_path]

            completed = subprocess.run(
                args,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                cwd=artifact_dir,
            )
            stderr = completed.stderr.strip()
            if completed.returncode != 0:
                raise CudaVerifierError(
                    op_name,
                    "verifier process returned a failure status",
                    returncode=completed.returncode,
                    stderr=stderr,
                )

            final_shape = tuple(target_shape) if target_shape is not None else cuda_inputs[0].shape
            output = _read_array_output(
                op_name,
                output_path,
                output_dtype,
                final_shape,
                "output",
                stderr,
            )
            if not sidecar_specs:
                return output

            parsed_sidecars = {}
            for spec in sidecar_specs:
                sidecar_path = os.path.abspath(os.path.join(artifact_dir, spec.name))
                parsed_sidecars[spec.name] = _read_array_output(
                    op_name,
                    sidecar_path,
                    spec.dtype,
                    tuple(spec.shape),
                    f"sidecar {spec.name}",
                    stderr,
                )
            return CudaRunResult(output=output, sidecars=parsed_sidecars)
    except CudaVerifierError:
        raise
    except (OSError, ValueError) as exc:
        raise CudaVerifierError(op_name, str(exc)) from exc
