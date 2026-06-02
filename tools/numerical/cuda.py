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

import numpy as np


CUDA_VERIFY_DIR = os.environ.get("CUDA_VERIFY_DIR", "cache")


def run_cuda_ground_truth(op_name, inputs_f32, params_binary=None, output_dtype=np.float32, target_shape=None):
    exe = os.path.join(CUDA_VERIFY_DIR, f"verify_{op_name}")
    if not os.path.exists(exe):
        print(f"⚠️  Missing CUDA executable: {exe}")
        return None
        
    cuda_inputs = list(inputs_f32) # Copy list

    files = []
    for i, arr in enumerate(cuda_inputs):
        if arr is None:
            files.append("null")
            continue
        fname = f"tmp_in_{i}.bin"
        #arr.tofile(fname)
        np.ascontiguousarray(arr).tofile(fname)
        files.append(fname)
    
    if params_binary is not None:
        p_fname = "tmp_params.bin"
        with open(p_fname, "wb") as f:
            f.write(params_binary)
        files.append(p_fname)

    out_fname = "tmp_out.bin"
    
    try:
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

        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        final_shape = target_shape if target_shape is not None else cuda_inputs[0].shape
        result = np.fromfile(out_fname, dtype=output_dtype).reshape(final_shape)
        
    except Exception as e:
        print(f"CUDA Fail [{op_name}]: {e}") 
        result = None
    # except subprocess.CalledProcessError as e:
    #     msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
    #     print(f"CUDA Fail [{op_name}]: {msg}")
    #     result = None
    finally:
        for f in files:
            if f != "null" and os.path.exists(f): os.remove(f)
        if os.path.exists(out_fname): os.remove(out_fname)
            
    return result
