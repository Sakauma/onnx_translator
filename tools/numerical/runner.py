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

import traceback

import numpy as np

from nn import Tensor

from .compare import INTEGER_DTYPES, check_accuracy, compare_integer_output
from .cuda import CudaSidecarSpec, run_cuda_ground_truth
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
    """执行一个算子验证计划，并汇总 C 后端与 CUDA 参考实现的误差。

    ``shapes``、``dtypes`` 与算子输入位置一一对应；可选输入使用 ``None``
    保留位置。函数每轮重新生成样本，先执行 Python/C runtime，再使用相同样本
    调用 CUDA verifier。返回值依次为绝对误差样本、相对误差样本和整项是否通过。

    多输出算子会委托给 ``handle_special_output``，普通单输出算子继续走本函数
    的统一转换和比较路径。任何一轮失败都会停止当前计划，避免后续样本掩盖首个错误。
    """
    if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    init_args = init_args or {}
    print(f"🧪 Testing {op_name.upper()}: {dtypes} -> {out_dtype}")

    # 容差和 CUDA 数据通路按算子族集中解析，调用方不应再维护平行白名单。
    verification_config = resolve_verification_config(op_name, out_dtype)
    atol, rtol = verification_config.atol, verification_config.rtol

    pass_cnt = 0
    stats_abs = []
    stats_rel = []
    integer_failure_reason = None

    for i in range(iterations):
        # 每轮都重新生成样本，保证 iterations 真正覆盖不同输入，而非重复比较同一缓冲区。
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
            
        # params.bin 是 CUDA verifier 的稳定参数协议，必须基于本轮实际输入和 C 输出构造。
        params_bin = build_cuda_params(op_name, inputs_np, init_args, shapes, dtypes, out_dtype, nps_out)

        # 多输出和 sidecar 协议在这里截获；NOT_HANDLED 才进入普通单输出路径。
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

        # 标量统一物化成单元素数组，因为 CUDA 文件协议只传输张量缓冲区。
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

        sidecar_specs = None
        if op_name == "topk":
            sidecar_specs = [CudaSidecarSpec("tmp_out_idx.bin", np.int64, expected_shape)]
        cuda_result = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=out_np_dtype,
            target_shape=expected_shape,
            sidecars=sidecar_specs,
        )

        if cuda_result is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")
        cuda_out = cuda_result.output if sidecar_specs else cuda_result

        # TopK 的 values 走主输出文件，indices 由 verifier 写入独立 sidecar。
        if op_name == "topk":
            cuda_topk_indices = cuda_result.sidecars["tmp_out_idx.bin"]

            if out_dtype in INTEGER_DTYPES:
                ok_vals, fail_mask, _nps_vals, cuda_out, value_reason = compare_integer_output(
                    nps_out,
                    cuda_out,
                    out_dtype,
                )
                max_abs = 0.0 if ok_vals else -1.0
                max_rel = 0.0 if ok_vals else -1.0
            else:
                nps_vals = to_float32(nps_out, out_dtype)
                ok_vals, max_abs, max_rel, fail_mask = check_accuracy(
                    nps_vals,
                    cuda_out,
                    atol,
                    rtol,
                    out_dtype,
                )
                value_reason = None

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
                elif value_reason is not None:
                    print(f"     TopK integer values mismatch: {value_reason}")
                else:
                    print(f"     Max Abs Diff: {max_abs:.6f} (Limit: {atol})")
                    print(f"     Max Rel Diff: {max_rel:.6f} (Limit: {rtol})")
                break
            continue

        if out_dtype == "bool":
            cuda_out = cuda_out.astype(np.float32)   

        # BitCast 比较原始字节，整数比较精确值，其余 dtype 量化回声明精度后比较数值误差。
        if op_name == "bitcast":
            nps_raw = np.ascontiguousarray(nps_out).view(np.uint8)
            cuda_raw = np.ascontiguousarray(cuda_out).view(np.uint8)
            is_ok = np.array_equal(nps_raw, cuda_raw)
            max_abs = 0.0 if is_ok else -1.0
            max_rel = 0.0 if is_ok else -1.0
            fail_mask = None if is_ok else (nps_raw != cuda_raw)
            nps_f32 = np.asarray(nps_out).reshape(expected_shape)
        elif out_dtype in INTEGER_DTYPES:
            is_ok, fail_mask, nps_int, cuda_out, integer_failure_reason = compare_integer_output(
                nps_out,
                cuda_out,
                out_dtype,
            )
            max_abs = 0.0 if is_ok else -1.0
            max_rel = 0.0 if is_ok else -1.0
            nps_f32 = nps_int
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
            elif integer_failure_reason is not None: print(f"     Integer comparison failed: {integer_failure_reason}")
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
                # 只对普通逐元素广播算子反查同坐标输入；复杂 kernel 的坐标映射并非一一对应。
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
