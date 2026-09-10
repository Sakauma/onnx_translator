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

import os
import subprocess
import tempfile
import threading
import shutil

import numpy as np


CUDA_VERIFY_DIR = os.environ.get("CUDA_VERIFY_DIR", "cache")
_artifact_state = threading.local()


def artifact_path(name):
    directory = getattr(_artifact_state, "directory", None)
    return os.path.join(directory, name) if directory else name


def cleanup_cuda_artifacts():
    directory = getattr(_artifact_state, "directory", None)
    if directory:
        shutil.rmtree(directory, ignore_errors=True)
        _artifact_state.directory = None


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


def run_cuda_ground_truth(op_name, inputs_f32, params_binary=None, output_dtype=np.float32, target_shape=None, keep_artifacts=False):
    exe = os.path.abspath(os.path.join(CUDA_VERIFY_DIR, f"verify_{op_name}"))
    if not os.path.exists(exe):
        raise CudaVerifierError(op_name, f"missing executable: {exe}")
        
    cuda_inputs = list(inputs_f32) # Copy list
    cleanup_cuda_artifacts()
    artifact_dir = tempfile.mkdtemp(prefix="onnx-translator-cuda-")
    _artifact_state.directory = artifact_dir

    files = []
    out_fname = artifact_path("tmp_out.bin")
    try:
        for i, arr in enumerate(cuda_inputs):
            if arr is None:
                files.append("null")
                continue
            fname = artifact_path(f"tmp_in_{i}.bin")
            _write_array_input(op_name, fname, arr)
            files.append(fname)
        if params_binary is not None:
            p_fname = artifact_path("tmp_params.bin")
            _write_params_input(op_name, p_fname, params_binary)
            files.append(p_fname)
        # args = [exe, str(cuda_inputs[0].size)] + files + [out_fname]
        # if target_shape is not None:
        #      out_elem_count = int(np.prod(target_shape))
        #      args[1] = str(out_elem_count)
        out_elem_count = int(np.prod(target_shape)) if target_shape is not None else int(cuda_inputs[0].size)

        if op_name == "resize":
            x_file = files[0]
            p_file = files[-1] if params_binary is not None else None
            if p_file is None:
                raise RuntimeError("resize requires params_binary")
            args = [exe, str(out_elem_count), x_file, p_file, out_fname]
        else:
            args = [exe, str(out_elem_count)] + files + [out_fname]

        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=artifact_dir,
        )
        if completed.returncode != 0:
            raise CudaVerifierError(
                op_name,
                "verifier process returned a failure status",
                returncode=completed.returncode,
                stderr=completed.stderr.strip(),
            )
        if not os.path.exists(out_fname):
            raise CudaVerifierError(op_name, f"verifier did not create output: {out_fname}")
        final_shape = target_shape if target_shape is not None else cuda_inputs[0].shape
        expected_count = int(np.prod(final_shape))
        expected_bytes = expected_count * np.dtype(output_dtype).itemsize
        actual_bytes = os.path.getsize(out_fname)
        if actual_bytes != expected_bytes:
            raise CudaVerifierError(
                op_name,
                f"invalid output size: expected {expected_bytes} bytes, got {actual_bytes}",
            )
        result = np.fromfile(out_fname, dtype=output_dtype)
        if result.size != expected_count:
            raise CudaVerifierError(
                op_name,
                f"invalid output element count: expected {expected_count}, got {result.size}",
            )
        result = result.reshape(final_shape)
    except CudaVerifierError:
        raise
    except (OSError, ValueError) as exc:
        raise CudaVerifierError(op_name, str(exc)) from exc
    finally:
        for f in files:
            if f != "null" and os.path.exists(f): os.remove(f)
        if os.path.exists(out_fname): os.remove(out_fname)
        if not keep_artifacts:
            cleanup_cuda_artifacts()
            
    return result
