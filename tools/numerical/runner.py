# /**
#   ******************************************************************************
#   * @file        runner.py
#   * @author      Egor Izmaylov
#   * @brief       执行单个算子的数值验证计划，包括输入准备、参数打包和结果比较。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import os
import traceback

import numpy as np

from nn import Tensor

from .compare import check_accuracy
from .cuda import run_cuda_ground_truth
from .dtype import quantize_to_dtype_float32, to_float32
from .runner_config import resolve_verification_config
from .runner_cuda_inputs import build_cuda_inputs, resolve_cuda_output_dtype
from .runner_cuda_params import build_cuda_params
from .runner_inputs import prepare_input_samples
from .runner_nps import run_nps_forward
from .runner_special_outputs import (
    SpecialOutputAction,
    SpecialOutputState,
    handle_special_output,
)


def verify_op(op_cls, op_name, shapes, dtypes, out_dtype, init_args=None, iterations=5):
    init_args = init_args or {}
    print(f"🧪 Testing {op_name.upper()}: {dtypes} -> {out_dtype}")
    
    verification_config = resolve_verification_config(op_name, out_dtype)
    atol, rtol = verification_config.atol, verification_config.rtol

    pass_cnt = 0
    stats_abs = []
    stats_rel = []
    
    for i in range(iterations):
        inputs_np = prepare_input_samples(op_name, shapes, dtypes, init_args)
        inputs_tensor = []
        for data, d in zip(inputs_np, dtypes):
            if data is not None: inputs_tensor.append(Tensor(*data.shape, dtype=d, data=data))
            else: inputs_tensor.append(None)

        try:
            nps_result = run_nps_forward(op_cls, op_name, inputs_tensor, init_args, out_dtype)
            nps_out = nps_result.output
            nps_topk_indices = nps_result.topk_indices
        except Exception as e:
            print(f"  ❌ Iter {i} Crash: {e}")
            traceback.print_exc()
            continue
            
        params_bin = build_cuda_params(op_name, inputs_np, init_args, shapes, dtypes, out_dtype, nps_out)

        special_state = SpecialOutputState(
            op_cls=op_cls,
            op_name=op_name,
            inputs_np=inputs_np,
            dtypes=dtypes,
            out_dtype=out_dtype,
            init_args=init_args,
            params_bin=params_bin,
            nps_out=nps_out,
            atol=atol,
            rtol=rtol,
            iteration=i,
            pass_count=pass_cnt,
            stats_abs=stats_abs,
            stats_rel=stats_rel,
        )
        special_action = handle_special_output(special_state)
        pass_cnt = special_state.pass_count
        if special_action is SpecialOutputAction.STOP:
            break
        if special_action is SpecialOutputAction.CONTINUE:
            continue

        # 4. 数据转换与广播处理
        expected_shape = nps_out.shape
        if expected_shape == ():
            expected_shape = (1,)  # 统一当成 1 元素张量来跑 CUDA/读写 bin
            nps_out = np.array([nps_out], dtype=nps_out.dtype)
        cuda_inputs = build_cuda_inputs(
            op_name,
            inputs_np,
            dtypes,
            init_args,
            expected_shape,
            verification_config,
        )
        out_np_dtype = resolve_cuda_output_dtype(op_name, out_dtype, verification_config)

        cuda_out = run_cuda_ground_truth(
        op_name,
        cuda_inputs,
        params_binary=params_bin,
        output_dtype=out_np_dtype,
        target_shape=expected_shape
        )

        if cuda_out is None:
            continue

        if op_name == "topk":
            idx_path = "tmp_out_idx.bin"
            if not os.path.exists(idx_path):
                print(f"  ❌ Iter {i} FAILED")
                print("     Missing tmp_out_idx.bin for TopK")
                break

            cuda_topk_indices = np.fromfile(idx_path, dtype=np.int64).reshape(expected_shape)
            os.remove(idx_path)

            nps_vals = to_float32(nps_out, out_dtype)
            ok_vals, max_abs, max_rel, fail_mask = check_accuracy(nps_vals, cuda_out, atol, rtol, out_dtype)

            ok_idx = np.array_equal(
                np.asarray(nps_topk_indices).astype(np.int64),
                np.asarray(cuda_topk_indices).astype(np.int64)
            )

            if max_abs >= 0:
                stats_abs.append(max_abs)
                stats_rel.append(max_rel)

            if ok_vals and ok_idx:
                pass_cnt += 1
            else:
                print(f"  ❌ Iter {i} FAILED")
                if not ok_idx:
                    print("     TopK indices mismatch")
                else:
                    print(f"     Max Abs Diff: {max_abs:.6f} (Limit: {atol})")
                    print(f"     Max Rel Diff: {max_rel:.6f} (Limit: {rtol})")
                break
            continue

        if out_dtype == "bool":
            cuda_out = cuda_out.astype(np.float32)   

        # 6. 对比
        # nps_f32 = to_float32(nps_out, out_dtype)
        # is_ok, max_abs, max_rel, fail_mask = check_accuracy(nps_f32, cuda_out, atol, rtol, out_dtype)
        if op_name == "bitcast":
            nps_raw = np.ascontiguousarray(nps_out).view(np.uint8)
            cuda_raw = np.ascontiguousarray(cuda_out).view(np.uint8)
            is_ok = np.array_equal(nps_raw, cuda_raw)
            max_abs = 0.0 if is_ok else -1.0
            max_rel = 0.0 if is_ok else -1.0
            fail_mask = None if is_ok else (nps_raw != cuda_raw)
            nps_f32 = np.asarray(nps_out).reshape(expected_shape)
        elif out_dtype in {"int32", "int64"}:
            int_dtype = np.int32 if out_dtype == "int32" else np.int64
            nps_int = np.asarray(nps_out).astype(int_dtype)
            cuda_int = np.asarray(cuda_out).astype(int_dtype)
            is_ok = np.array_equal(nps_int, cuda_int)
            max_abs = 0.0 if is_ok else -1.0
            max_rel = 0.0 if is_ok else -1.0
            fail_mask = None if is_ok else (nps_int != cuda_int)
            nps_f32 = nps_int.astype(np.float32)
        else:
            nps_f32 = to_float32(nps_out, out_dtype)
            cuda_out = quantize_to_dtype_float32(cuda_out, out_dtype)
            is_ok, max_abs, max_rel, fail_mask = check_accuracy(nps_f32, cuda_out, atol, rtol, out_dtype)
        
        if max_abs >= 0:
            stats_abs.append(max_abs)
            stats_rel.append(max_rel)
        if is_ok:
            pass_cnt += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            if max_abs == -999.0: print(f"     Failed due to Overflow/Inf Logic Mismatch")
            elif max_abs == -1.0: print(f"     Failed due to NaN/Inf Mismatch")
            else:
                print(f"     Max Abs Diff: {max_abs:.6f} (Limit: {atol})")
                print(f"     Max Rel Diff: {max_rel:.6f} (Limit: {rtol})")
            
            if fail_mask is not None and np.any(fail_mask):
                idx_flat = np.argmax(fail_mask)
                idx = np.unravel_index(idx_flat, fail_mask.shape)
                print(f"     🔍 Debug Sample at {idx}:")
                print(f"        GT (CUDA) = {cuda_out[idx]}")
                print(f"        NPS (C)   = {nps_f32[idx]}")
                # 显示原始输入值
                for k, inp_arr in enumerate(inputs_np):
                    val_disp = ""
                    if inp_arr is None: val_disp = "None"
                    else:
                        try:
                            if not verification_config.complex_kernel and verification_config.broadcast_inputs:
                                val_disp = np.broadcast_to(inp_arr, expected_shape)[idx]
                            else:
                                if inp_arr.shape == expected_shape:
                                    val_disp = inp_arr[idx]
                                else:
                                    val_disp = f"Shape{inp_arr.shape} (No direct mapping)"
                        except Exception as e:
                            val_disp = f"Error: {e}"
                            
                    print(f"        Input {k}   = {val_disp}")
            break

    if pass_cnt == iterations:
        print(f"  ✅ Pass ({pass_cnt}/{iterations})\n")
    else:
        print(f"  ⚠️  Fail\n")
    return stats_abs, stats_rel, pass_cnt == iterations
