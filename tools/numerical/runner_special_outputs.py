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
import os

import numpy as np

import nn

from .compare import check_accuracy
from .cuda import run_cuda_ground_truth
from .dtype import quantize_to_dtype_float32, to_float32


class SpecialOutputAction(Enum):
    NOT_HANDLED = auto()
    CONTINUE = auto()
    STOP = auto()


@dataclass
class SpecialOutputState:
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

        cuda_y = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=y_np.shape,
        )
        if cuda_y is None:
            return SpecialOutputAction.CONTINUE

        missing_paths = [path for _name, _expected, path in side_specs if not os.path.exists(path)]
        if missing_paths:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Missing {op_cls.__name__} sidecar output: {', '.join(missing_paths)}")
            for _name, _expected, path in side_specs:
                if os.path.exists(path):
                    os.remove(path)
            return SpecialOutputAction.STOP

        comparisons = [("Y", y_np, cuda_y)]
        for name, expected, path in side_specs:
            cuda_side = np.fromfile(path, dtype=np.float64).reshape(expected.shape)
            os.remove(path)
            comparisons.append((name, expected, cuda_side))

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

    if op_name == "dropout":
        y_np, mask_np = [np.asarray(out) for out in nps_out]
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32)),
        ]
        cuda_y = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float32,
            target_shape=y_np.shape,
        )
        if cuda_y is None:
            return SpecialOutputAction.CONTINUE

        mask_path = "tmp_dropout_mask.bin"
        if not os.path.exists(mask_path):
            print(f"  ❌ Iter {i} FAILED")
            print("     Missing Dropout mask sidecar output")
            return SpecialOutputAction.STOP
        cuda_mask = np.fromfile(mask_path, dtype=np.uint8).reshape(mask_np.shape).astype(np.bool_)
        os.remove(mask_path)

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

    if op_name == "batch_normalization" and int(init_args.get("training_mode", 0)):
        y_np, running_mean_np, running_var_np = [np.asarray(out) for out in nps_out]
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[idx], dtypes[idx]).astype(np.float64))
            for idx in range(5)
        ]
        cuda_y = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=y_np.shape,
        )
        if cuda_y is None:
            return SpecialOutputAction.CONTINUE

        side_paths = {
            "running_mean": "tmp_batch_norm_running_mean.bin",
            "running_var": "tmp_batch_norm_running_var.bin",
        }
        if not all(os.path.exists(path) for path in side_paths.values()):
            print(f"  ❌ Iter {i} FAILED")
            print("     Missing BatchNormalization training sidecar output")
            for path in side_paths.values():
                if os.path.exists(path):
                    os.remove(path)
            return SpecialOutputAction.STOP

        cuda_running_mean = np.fromfile(side_paths["running_mean"], dtype=np.float64).reshape(running_mean_np.shape)
        cuda_running_var = np.fromfile(side_paths["running_var"], dtype=np.float64).reshape(running_var_np.shape)
        for path in side_paths.values():
            os.remove(path)

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

    if op_name == "layer_normalization" and int(init_args.get("emit_stats", 0)):
        y_np, mean_np, inv_std_np = [np.asarray(out) for out in nps_out]
        cuda_inputs = [
            np.ascontiguousarray(to_float32(inputs_np[idx], dtypes[idx]).astype(np.float64))
            for idx in range(3)
        ]
        cuda_y = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=y_np.shape,
        )
        if cuda_y is None:
            return SpecialOutputAction.CONTINUE

        side_paths = {
            "mean": "tmp_layer_norm_mean.bin",
            "inv_std": "tmp_layer_norm_inv_std.bin",
        }
        if not all(os.path.exists(path) for path in side_paths.values()):
            print(f"  ❌ Iter {i} FAILED")
            print("     Missing LayerNormalization stats sidecar output")
            for path in side_paths.values():
                if os.path.exists(path):
                    os.remove(path)
            return SpecialOutputAction.STOP

        cuda_mean = np.fromfile(side_paths["mean"], dtype=np.float64).reshape(mean_np.shape)
        cuda_inv_std = np.fromfile(side_paths["inv_std"], dtype=np.float64).reshape(inv_std_np.shape)
        for path in side_paths.values():
            os.remove(path)

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
        cuda_loss = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float64,
            target_shape=loss_shape,
        )
        if cuda_loss is None:
            return SpecialOutputAction.CONTINUE

        loss_ref = quantize_to_dtype_float32(cuda_loss, out_dtype)
        loss_nps = to_float32(loss_cmp, out_dtype)
        loss_ok, loss_abs, loss_rel, _loss_fail = check_accuracy(loss_nps, loss_ref, atol, rtol, out_dtype)

        log_ok = True
        log_abs = 0.0
        log_rel = 0.0
        if log_prob_np is not None:
            log_path = "tmp_out_log_prob.bin"
            if not os.path.exists(log_path):
                print(f"  ❌ Iter {i} FAILED")
                print("     Missing SoftmaxCrossEntropyLoss log_prob sidecar output")
                return SpecialOutputAction.STOP
            cuda_log = np.fromfile(log_path, dtype=np.float64).reshape(log_prob_np.shape)
            os.remove(log_path)
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

    if op_name == "dynamic_quantize_linear":
        y_np, scale_np, zp_np = nps_out
        y_np = np.asarray(y_np, dtype=np.uint8)
        scale_np = np.asarray(scale_np, dtype=np.float32).reshape(())
        zp_np = np.asarray(zp_np, dtype=np.uint8).reshape(())

        flat_len = int(y_np.size)
        cuda_inputs = [np.ascontiguousarray(to_float32(inputs_np[0], dtypes[0]).astype(np.float32))]
        cuda_out = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=np.float32,
            target_shape=(flat_len + 2,),
        )
        if cuda_out is None:
            return SpecialOutputAction.CONTINUE

        cuda_flat = np.asarray(cuda_out, dtype=np.float32).reshape(-1)
        cuda_y = np.rint(cuda_flat[:flat_len]).clip(0, 255).astype(np.uint8).reshape(y_np.shape)
        cuda_scale = np.asarray(cuda_flat[flat_len], dtype=np.float32).reshape(())
        cuda_zp = np.asarray(np.rint(cuda_flat[flat_len + 1]).clip(0, 255), dtype=np.uint8).reshape(())

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
            return SpecialOutputAction.CONTINUE

        cuda_flat = np.asarray(cuda_out, dtype=np.float32).reshape(-1)
        offset = 0
        ok_all = True
        max_abs_all = 0.0
        max_rel_all = 0.0
        failed_index = -1
        for out_idx, expected_piece in enumerate(flat_outputs):
            piece_len = int(expected_piece.size)
            cuda_piece = cuda_flat[offset:offset + piece_len].reshape(expected_piece.shape)
            offset += piece_len
            nps_piece = to_float32(expected_piece, out_dtype)
            cuda_piece = quantize_to_dtype_float32(cuda_piece, out_dtype)
            ok_piece, max_abs, max_rel, _fail_mask = check_accuracy(nps_piece, cuda_piece, atol, rtol, out_dtype)
            max_abs_all = max(max_abs_all, max_abs if max_abs >= 0 else 0.0)
            max_rel_all = max(max_rel_all, max_rel if max_rel >= 0 else 0.0)
            if not ok_piece:
                ok_all = False
                failed_index = out_idx
                break

        state.stats_abs.append(max_abs_all)
        state.stats_rel.append(max_rel_all)
        if ok_all:
            state.pass_count += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            print(f"     Split output {failed_index} mismatch")
            print(f"     Max Abs Diff: {max_abs_all:.6f} (Limit: {atol})")
            print(f"     Max Rel Diff: {max_rel_all:.6f} (Limit: {rtol})")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    if op_name == "unique":
        values_np, indices_np, inverse_np, counts_np = [np.asarray(out) for out in nps_out]
        input_arr = inputs_np[0]
        if dtypes[0] == "int64":
            cuda_inputs = [np.ascontiguousarray(input_arr.astype(np.int64))]
            cuda_value_dtype = np.int64
        else:
            cuda_inputs = [np.ascontiguousarray(to_float32(input_arr, dtypes[0]).astype(np.float32))]
            cuda_value_dtype = np.float32

        cuda_values = run_cuda_ground_truth(
            op_name,
            cuda_inputs,
            params_binary=params_bin,
            output_dtype=cuda_value_dtype,
            target_shape=values_np.shape,
        )
        if cuda_values is None:
            return SpecialOutputAction.CONTINUE

        side_paths = {
            "indices": "tmp_unique_indices.bin",
            "inverse": "tmp_unique_inverse.bin",
            "counts": "tmp_unique_counts.bin",
        }
        if not all(os.path.exists(path) for path in side_paths.values()):
            print(f"  ❌ Iter {i} FAILED")
            print("     Missing Unique sidecar output")
            for path in side_paths.values():
                if os.path.exists(path):
                    os.remove(path)
            return SpecialOutputAction.STOP

        cuda_indices = np.fromfile(side_paths["indices"], dtype=np.int64).reshape(indices_np.shape)
        cuda_inverse = np.fromfile(side_paths["inverse"], dtype=np.int64).reshape(inverse_np.shape)
        cuda_counts = np.fromfile(side_paths["counts"], dtype=np.int64).reshape(counts_np.shape)
        for path in side_paths.values():
            os.remove(path)

        if out_dtype == "int64":
            values_ok = np.array_equal(values_np.astype(np.int64), cuda_values.astype(np.int64))
            value_abs = 0.0 if values_ok else -1.0
            value_rel = 0.0 if values_ok else -1.0
        else:
            nps_values = to_float32(values_np, out_dtype)
            cuda_values = quantize_to_dtype_float32(cuda_values, out_dtype)
            values_ok, value_abs, value_rel, _fail_mask = check_accuracy(nps_values, cuda_values, atol, rtol, out_dtype)

        indices_ok = np.array_equal(indices_np.astype(np.int64), cuda_indices)
        inverse_ok = np.array_equal(inverse_np.astype(np.int64), cuda_inverse)
        counts_ok = np.array_equal(counts_np.astype(np.int64), cuda_counts)
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
            if not indices_ok:
                print("     Unique indices mismatch")
            if not inverse_ok:
                print("     Unique inverse mismatch")
            if not counts_ok:
                print("     Unique counts mismatch")
            return SpecialOutputAction.STOP
        return SpecialOutputAction.CONTINUE

    return SpecialOutputAction.NOT_HANDLED
