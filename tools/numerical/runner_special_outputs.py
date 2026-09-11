# /**
#   ******************************************************************************
#   * @file        runner_special_outputs.py
#   * @author      Egor Izmaylov
#   * @brief       处理数值验证中的多输出、sidecar 文件和专用比较协议。
#   * @details     2026.07.15  V1.0.0  从 runner.py 拆分特殊结果处理职责
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

import nn

from .compare import INTEGER_DTYPES, check_accuracy, compare_integer_output
from .cuda import CudaSidecarSpec, run_cuda_ground_truth
from .dtype import quantize_to_dtype_float32, to_float32
from .output_contracts import (
    OutputContractError,
    require_array,
    require_binary_uint8,
    require_packed_uint8_field,
    require_shape,
)


class SpecialOutputAction(Enum):
    """告知通用调度器当前迭代是否已被特殊协议消费。"""

    # 未命中特殊协议，继续执行 runner.py 的普通单输出路径。
    NOT_HANDLED = auto()
    # 本轮已完成或 CUDA 无输出，跳到下一轮。
    CONTINUE = auto()
    # 已发现确定失败，停止当前算子的剩余迭代。
    STOP = auto()


@dataclass
class SpecialOutputState:
    """特殊输出处理器共享的本轮上下文和累计统计状态。

    ``pass_count`` 会由处理器原地更新；误差列表与调用方共享，避免每个多输出
    协议重复封装统计返回值。其余字段视为只读。
    """

    op_cls: type
    op_name: str
    inputs_np: list
    dtypes: list[str]
    out_dtype: str
    init_args: dict
    params_bin: bytes
    nps_out: object
    atol: float
    rtol: float
    iteration: int
    pass_count: int
    stats_abs: list[float]
    stats_rel: list[float]


def handle_special_output(state):
    """执行多输出或 sidecar 算子的专用 CUDA 比较协议。

    返回 ``NOT_HANDLED`` 时不得修改累计状态。命中协议后必须返回 ``CONTINUE``
    或 ``STOP``，确保主调度器不会再次把列表输出当作普通 ndarray 处理。
    """
    op_cls = state.op_cls
    op_name = state.op_name
    inputs_np = state.inputs_np
    dtypes = state.dtypes
    out_dtype = state.out_dtype
    init_args = state.init_args
    params_bin = state.params_bin
    nps_out = state.nps_out
    atol = state.atol
    rtol = state.rtol
    i = state.iteration

    # 循环网络：主文件保存 Y，隐藏态和 cell state 通过 sidecar 返回。
    if op_name in {"rnn", "gru", "lstm"}:
        recurrent_outputs = [np.asarray(out) for out in nps_out]
        y_np = recurrent_outputs[0]
        side_specs = [("Y_h", recurrent_outputs[1], f"tmp_{op_name}_y_h.bin")]
        if op_name == "lstm":
            side_specs.append(("Y_c", recurrent_outputs[2], "tmp_lstm_y_c.bin"))

        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float64)),
            np.ascontiguousarray(to_float32(inputs_np[1], dtypes[1]).astype(np.float64)),
            np.ascontiguousarray(to_float32(inputs_np[2], dtypes[2]).astype(np.float64)),
            np.ascontiguousarray(to_float32(inputs_np[3], dtypes[3]).astype(np.float64)),
            np.ascontiguousarray(inputs_np[4].astype(np.int64)),
            np.ascontiguousarray(to_float32(inputs_np[5], dtypes[5]).astype(np.float64)),
        ]
        if op_name == "lstm":
            cuda_inputs.extend(
                [
                    np.ascontiguousarray(to_float32(inputs_np[6], dtypes[6]).astype(np.float64)),
                    np.ascontiguousarray(to_float32(inputs_np[7], dtypes[7]).astype(np.float64)),
                ]
            )

        cuda_result = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=y_np.shape,
            sidecars=[
                CudaSidecarSpec(path, np.float64, expected.shape)
                for _name, expected, path in side_specs
            ],
        )
        if cuda_result is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")
        comparisons = [("Y", y_np, cuda_result.output)]
        for name, expected, path in side_specs:
            comparisons.append((name, expected, cuda_result.sidecars[path]))

        ok_all = True
        max_abs_all = 0.0
        max_rel_all = 0.0
        failed_name = None
        for name, expected, cuda_value in comparisons:
            cuda_ref = quantize_to_dtype_float32(cuda_value, out_dtype)
            expected_cmp = to_float32(expected, out_dtype)
            ok, cur_abs, cur_rel, _fail = check_accuracy(expected_cmp, cuda_ref, atol, rtol, out_dtype)
            max_abs_all = max(max_abs_all, cur_abs if cur_abs >= 0 else 0.0)
            max_rel_all = max(max_rel_all, cur_rel if cur_rel >= 0 else 0.0)
            if not ok and failed_name is None:
                failed_name = name
            ok_all = ok_all and ok

        state.stats_abs.append(max_abs_all)
        state.stats_rel.append(max_rel_all)
        if ok_all:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     {op_cls.__name__} {failed_name} mismatch")
            print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
            print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    # Dropout 同时比较数值输出和 bool mask；仅比较 Y 会漏掉随机掩码协议错误。
    if op_name == "dropout":
        try:
            y_np = require_shape(
                nps_out[0],
                name="Dropout NPS y",
                shape=np.asarray(inputs_np[0]).shape,
            )
            mask_np = require_array(
                nps_out[1],
                name="Dropout NPS mask",
                dtype=np.bool_,
                shape=y_np.shape,
            )
        except OutputContractError as exc:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Output contract mismatch: {exc}")
            return SpecialOutputAction.STOP
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32)),
        ]
        cuda_result = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float32,
            target_shape=y_np.shape,
            sidecars=[CudaSidecarSpec("tmp_dropout_mask.bin", np.uint8, mask_np.shape)],
        )
        if cuda_result is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")
        try:
            cuda_y = require_shape(
                cuda_result.output, name="Dropout CUDA y", shape=y_np.shape
            )
            cuda_mask_wire = require_binary_uint8(
                cuda_result.sidecars["tmp_dropout_mask.bin"],
                name="Dropout CUDA mask sidecar",
                shape=mask_np.shape,
            )
        except OutputContractError as exc:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Output contract mismatch: {exc}")
            return SpecialOutputAction.STOP
        cuda_mask = cuda_mask_wire.astype(np.bool_)

        nps_y = to_float32(y_np, out_dtype)
        cuda_y = quantize_to_dtype_float32(cuda_y, out_dtype)
        y_ok, max_abs, max_rel, _fail_mask = check_accuracy(nps_y, cuda_y, atol, rtol, out_dtype)
        mask_ok = np.array_equal(mask_np.astype(np.bool_), cuda_mask)

        state.stats_abs.append(max_abs if max_abs >= 0 else 0.0)
        state.stats_rel.append(max_rel if max_rel >= 0 else 0.0)
        if y_ok and mask_ok:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            if not y_ok:
                print(f"     Dropout y mismatch: Max Abs Diff {max_abs:.6f}, Max Rel Diff {max_rel:.6f}")
            if not mask_ok:
                print("     Dropout mask mismatch")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    # 训练模式额外返回更新后的 running mean/variance，推理模式仍走普通路径。
    if op_name == "batch_normalization" and int(init_args.get("training_mode", 0)):
        y_np, running_mean_np, running_var_np = [np.asarray(out) for out in nps_out]
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[idx], dtypes[idx]).astype(np.float64))
            for idx in range(5)
        ]
        cuda_result = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=y_np.shape,
            sidecars=[
                CudaSidecarSpec("tmp_batch_norm_running_mean.bin", np.float64, running_mean_np.shape),
                CudaSidecarSpec("tmp_batch_norm_running_var.bin", np.float64, running_var_np.shape),
            ],
        )
        if cuda_result is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")
        cuda_y = cuda_result.output
        cuda_running_mean = cuda_result.sidecars["tmp_batch_norm_running_mean.bin"]
        cuda_running_var = cuda_result.sidecars["tmp_batch_norm_running_var.bin"]

        comparisons = [
            ("y", y_np, cuda_y),
            ("running_mean", running_mean_np, cuda_running_mean),
            ("running_var", running_var_np, cuda_running_var),
        ]
        ok_all = True
        max_abs_all = 0.0
        max_rel_all = 0.0
        failed_name = ""
        for name, expected, actual in comparisons:
            expected_f32 = to_float32(expected, out_dtype)
            actual_q = quantize_to_dtype_float32(actual, out_dtype)
            ok, max_abs, max_rel, _fail_mask = check_accuracy(expected_f32, actual_q, atol, rtol, out_dtype)
            max_abs_all = max(max_abs_all, max_abs if max_abs >= 0 else 0.0)
            max_rel_all = max(max_rel_all, max_rel if max_rel >= 0 else 0.0)
            if not ok:
                ok_all = False
                failed_name = name
                break

        state.stats_abs.append(max_abs_all)
        state.stats_rel.append(max_rel_all)
        if ok_all:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     BatchNormalization training {failed_name} mismatch")
            print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
            print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    # emit_stats 输出使用 stash_type 精度，不能统一按主输出 dtype 量化。
    if op_name == "layer_normalization" and int(init_args.get("emit_stats", 0)):
        y_np, mean_np, inv_std_np = [np.asarray(out) for out in nps_out]
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[idx], dtypes[idx]).astype(np.float64))
            for idx in range(3)
        ]
        cuda_result = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=y_np.shape,
            sidecars=[
                CudaSidecarSpec("tmp_layer_norm_mean.bin", np.float64, mean_np.shape),
                CudaSidecarSpec("tmp_layer_norm_inv_std.bin", np.float64, inv_std_np.shape),
            ],
        )
        if cuda_result is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")

        cuda_y = cuda_result.output
        cuda_mean = cuda_result.sidecars["tmp_layer_norm_mean.bin"]
        cuda_inv_std = cuda_result.sidecars["tmp_layer_norm_inv_std.bin"]

        stash_dtype = nn.onnx_dtype_mapping.get(int(init_args.get("stash_type", 1)), "float32")
        comparisons = [
            ("y", y_np, cuda_y, out_dtype),
            ("mean", mean_np, cuda_mean, stash_dtype),
            ("inv_std", inv_std_np, cuda_inv_std, stash_dtype),
        ]
        ok_all = True
        max_abs_all = 0.0
        max_rel_all = 0.0
        failed_name = ""
        for name, expected, actual, dtype_name in comparisons:
            expected_f32 = to_float32(expected, dtype_name)
            actual_q = quantize_to_dtype_float32(actual, dtype_name)
            ok, max_abs, max_rel, _fail_mask = check_accuracy(expected_f32, actual_q, atol, rtol, dtype_name)
            max_abs_all = max(max_abs_all, max_abs if max_abs >= 0 else 0.0)
            max_rel_all = max(max_rel_all, max_rel if max_rel >= 0 else 0.0)
            if not ok:
                ok_all = False
                failed_name = name
                break

        state.stats_abs.append(max_abs_all)
        state.stats_rel.append(max_rel_all)
        if ok_all:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     LayerNormalization {failed_name} mismatch")
            print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
            print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    # log_prob 是可选输出；存在时由固定 sidecar 返回并与 loss 分别比较。
    if op_name == "softmax_cross_entropy_loss":
        if isinstance(nps_out, list):
            loss_np = np.asarray(nps_out[0])
            log_prob_np = np.asarray(nps_out[1])
        else:
            loss_np = np.asarray(nps_out)
            log_prob_np = None
        loss_shape = loss_np.shape if loss_np.shape != () else (1,)
        loss_cmp = loss_np.reshape(loss_shape)
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float64)),
            np.ascontiguousarray(inputs_np[1].astype(np.int64)),
            None if len(inputs_np) <= 2 or inputs_np[2] is None else np.ascontiguousarray(to_float32(inputs_np[2], dtypes[2]).astype(np.float64)),
        ]
        sidecar_specs = []
        if log_prob_np is not None:
            sidecar_specs.append(
                CudaSidecarSpec("tmp_out_log_prob.bin", np.float64, log_prob_np.shape)
            )
        cuda_result = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=loss_shape,
            sidecars=sidecar_specs or None,
        )
        if cuda_result is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")
        cuda_loss = cuda_result.output if sidecar_specs else cuda_result

        loss_ref = quantize_to_dtype_float32(cuda_loss, out_dtype)
        loss_nps = to_float32(loss_cmp, out_dtype)
        loss_ok, loss_abs, loss_rel, _loss_fail = check_accuracy(loss_nps, loss_ref, atol, rtol, out_dtype)

        log_ok = True
        log_abs = 0.0
        log_rel = 0.0
        if log_prob_np is not None:
            cuda_log = cuda_result.sidecars["tmp_out_log_prob.bin"]
            log_ref = quantize_to_dtype_float32(cuda_log, out_dtype)
            log_nps = to_float32(log_prob_np, out_dtype)
            log_ok, log_abs, log_rel, _log_fail = check_accuracy(log_nps, log_ref, atol, rtol, out_dtype)

        max_abs = max(loss_abs if loss_abs >= 0 else 0.0, log_abs if log_abs >= 0 else 0.0)
        max_rel = max(loss_rel if loss_rel >= 0 else 0.0, log_rel if log_rel >= 0 else 0.0)
        state.stats_abs.append(max_abs)
        state.stats_rel.append(max_rel)
        if loss_ok and log_ok:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            if not loss_ok:
                print(f"     SCE loss mismatch: Max Abs Diff {loss_abs:.6f}, Max Rel Diff {loss_rel:.6f}")
            if not log_ok:
                print(f"     SCE log_prob mismatch: Max Abs Diff {log_abs:.6f}, Max Rel Diff {log_rel:.6f}")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    # CUDA 将 y、scale、zero_point 打包进一个 float32 文件，读取后需按段恢复类型。
    if op_name == "dynamic_quantize_linear":
        flat_len = int(np.asarray(nps_out[0]).size)
        cuda_inputs = [np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32))]
        cuda_out = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float32,
            target_shape=(flat_len + 2,),
        )
        if cuda_out is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")

        try:
            y_np = require_array(
                nps_out[0],
                name="DynamicQuantizeLinear NPS y",
                dtype=np.uint8,
                shape=np.asarray(inputs_np[0]).shape,
            )
            scale_np = require_array(
                nps_out[1],
                name="DynamicQuantizeLinear NPS y_scale",
                dtype=np.float32,
                shape=(),
                finite=True,
            )
            zp_np = require_array(
                nps_out[2],
                name="DynamicQuantizeLinear NPS y_zero_point",
                dtype=np.uint8,
                shape=(),
            )
            cuda_wire = require_array(
                cuda_out,
                name="DynamicQuantizeLinear CUDA packed output",
                dtype=np.float32,
                shape=(flat_len + 2,),
                finite=True,
            )
            cuda_flat = cuda_wire.reshape(-1)
            cuda_y_wire = require_packed_uint8_field(
                cuda_flat[:flat_len], name="DynamicQuantizeLinear CUDA y field"
            )
            cuda_zp_wire = require_packed_uint8_field(
                cuda_flat[flat_len + 1],
                name="DynamicQuantizeLinear CUDA y_zero_point field",
            )
        except OutputContractError as exc:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Output contract mismatch: {exc}")
            return SpecialOutputAction.STOP
        cuda_y = cuda_y_wire.astype(np.uint8).reshape(y_np.shape)
        cuda_scale = cuda_flat[flat_len].reshape(())
        cuda_zp = np.asarray(cuda_zp_wire, dtype=np.uint8).reshape(())

        y_ok = np.array_equal(y_np, cuda_y)
        scale_abs = float(abs(float(scale_np) - float(cuda_scale)))
        scale_rel = scale_abs / max(abs(float(cuda_scale)), 1e-12)
        scale_ok = scale_abs <= 1e-7 + 1e-6 * abs(float(cuda_scale))
        zp_ok = int(zp_np) == int(cuda_zp)

        y_abs = float(np.max(np.abs(y_np.astype(np.int16) - cuda_y.astype(np.int16)))) if y_np.size else 0.0
        zp_abs = float(abs(int(zp_np) - int(cuda_zp)))
        max_abs = max(y_abs, scale_abs, zp_abs)
        max_rel = scale_rel
        state.stats_abs.append(max_abs)
        state.stats_rel.append(max_rel)

        if y_ok and scale_ok and zp_ok:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            if not y_ok:
                print(f"     y mismatch, max uint8 diff: {y_abs:.0f}")
            if not scale_ok:
                print(f"     y_scale mismatch: CUDA={float(cuda_scale):.9g}, C={float(scale_np):.9g}")
            if not zp_ok:
                print(f"     y_zero_point mismatch: CUDA={int(cuda_zp)}, C={int(zp_np)}")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    # Split 将可变数量输出顺序拼接，按 C 输出 shape 切片还原后逐块比较。
    if op_name == "split":
        flat_outputs = [np.asarray(out) for out in nps_out]
        flat_len = int(sum(out.size for out in flat_outputs))
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32)),
        ]
        if len(inputs_np) > 1:
            cuda_inputs.append(np.ascontiguousarray(inputs_np[1].astype(np.int64)))
        cuda_out = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float32,
            target_shape=(flat_len,),
        )
        if cuda_out is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")

        cuda_flat = np.asarray(cuda_out, dtype=np.float32).reshape(-1)
        offset = 0
        ok_all = True
        max_abs_all = 0.0
        max_rel_all = 0.0
        failed_index = -1
        failed_reason = None
        for out_idx, expected_piece in enumerate(flat_outputs):
            piece_len = int(expected_piece.size)
            cuda_piece = cuda_flat[offset:offset + piece_len].reshape(expected_piece.shape)
            offset += piece_len
            if out_dtype in INTEGER_DTYPES:
                ok_piece, _fail_mask, _nps_int, _cuda_int, reason = compare_integer_output(
                    expected_piece,
                    cuda_piece,
                    out_dtype,
                )
                max_abs = 0.0 if ok_piece else -1.0
                max_rel = 0.0 if ok_piece else -1.0
            else:
                nps_piece = to_float32(expected_piece, out_dtype)
                cuda_piece = quantize_to_dtype_float32(cuda_piece, out_dtype)
                ok_piece, max_abs, max_rel, _fail_mask = check_accuracy(
                    nps_piece, cuda_piece, atol, rtol, out_dtype
                )
                reason = None
            max_abs_all = max(max_abs_all, max_abs if max_abs >= 0 else 0.0)
            max_rel_all = max(max_rel_all, max_rel if max_rel >= 0 else 0.0)
            if not ok_piece:
                ok_all = False
                failed_index = out_idx
                failed_reason = reason
                break

        state.stats_abs.append(max_abs_all)
        state.stats_rel.append(max_rel_all)
        if ok_all:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Split output {failed_index} mismatch")
            if failed_reason is not None:
                print(f"     Integer comparison failed: {failed_reason}")
            print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
            print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    # Unique 主文件保存 values，其余三个 int64 输出使用独立 sidecar。
    if op_name == "unique":
        values_np = np.asarray(nps_out[0])
        axis = init_args.get("axis")
        unique_len = values_np.size if axis is None else values_np.shape[int(axis) % values_np.ndim]
        inverse_shape = np.asarray(inputs_np[0]).shape if axis is None else (np.asarray(inputs_np[0]).shape[int(axis) % np.asarray(inputs_np[0]).ndim],)
        try:
            indices_np = require_array(
                nps_out[1], name="Unique NPS indices", dtype=np.int64, shape=(unique_len,)
            )
            inverse_np = require_array(
                nps_out[2], name="Unique NPS inverse_indices", dtype=np.int64, shape=inverse_shape
            )
            counts_np = require_array(
                nps_out[3], name="Unique NPS counts", dtype=np.int64, shape=(unique_len,)
            )
        except OutputContractError as exc:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Output contract mismatch: {exc}")
            return SpecialOutputAction.STOP
        input_arr = inputs_np[0]
        if dtypes[0] == "int64":
            cuda_inputs = [np.ascontiguousarray(input_arr.astype(np.int64))]
            cuda_value_dtype = np.int64
        else:
            cuda_inputs = [np.ascontiguousarray(to_float32(input_arr, dtypes[0]).astype(np.float32))]
            cuda_value_dtype = np.float32

        cuda_result = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=cuda_value_dtype,
            target_shape=values_np.shape,
            sidecars=[
                CudaSidecarSpec("tmp_unique_indices.bin", np.int64, indices_np.shape),
                CudaSidecarSpec("tmp_unique_inverse.bin", np.int64, inverse_np.shape),
                CudaSidecarSpec("tmp_unique_counts.bin", np.int64, counts_np.shape),
            ],
        )
        if cuda_result is None:
            raise RuntimeError(f"CUDA verifier produced no output [{op_name}]")
        cuda_values = cuda_result.output
        cuda_indices = cuda_result.sidecars["tmp_unique_indices.bin"]
        cuda_inverse = cuda_result.sidecars["tmp_unique_inverse.bin"]
        cuda_counts = cuda_result.sidecars["tmp_unique_counts.bin"]

        try:
            cuda_indices = require_array(
                cuda_indices, name="Unique CUDA indices sidecar", dtype=np.int64, shape=indices_np.shape
            )
            cuda_inverse = require_array(
                cuda_inverse, name="Unique CUDA inverse sidecar", dtype=np.int64, shape=inverse_np.shape
            )
            cuda_counts = require_array(
                cuda_counts, name="Unique CUDA counts sidecar", dtype=np.int64, shape=counts_np.shape
            )
        except OutputContractError as exc:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Output contract mismatch: {exc}")
            return SpecialOutputAction.STOP

        value_reason = None
        if out_dtype in INTEGER_DTYPES:
            values_ok, _fail_mask, _nps_int, _cuda_int, value_reason = compare_integer_output(
                values_np,
                cuda_values,
                out_dtype,
            )
            value_abs = 0.0 if values_ok else -1.0
            value_rel = 0.0 if values_ok else -1.0
        else:
            nps_values = to_float32(values_np, out_dtype)
            cuda_values = quantize_to_dtype_float32(cuda_values, out_dtype)
            values_ok, value_abs, value_rel, _fail_mask = check_accuracy(nps_values, cuda_values, atol, rtol, out_dtype)

        indices_ok = np.array_equal(indices_np, cuda_indices)
        inverse_ok = np.array_equal(inverse_np, cuda_inverse)
        counts_ok = np.array_equal(counts_np, cuda_counts)
        max_abs = value_abs if value_abs >= 0 else 0.0
        max_rel = value_rel if value_rel >= 0 else 0.0
        state.stats_abs.append(max_abs)
        state.stats_rel.append(max_rel)

        if values_ok and indices_ok and inverse_ok and counts_ok:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            if not values_ok:
                print("     Unique values mismatch")
                if value_reason is not None:
                    print(f"     Integer comparison failed: {value_reason}")
            if not indices_ok:
                print("     Unique indices mismatch")
            if not inverse_ok:
                print("     Unique inverse mismatch")
            if not counts_ok:
                print("     Unique counts mismatch")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    return SpecialOutputAction.NOT_HANDLED
