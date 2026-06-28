# /**
#   ******************************************************************************
#   * @file        runner_nps.py
#   * @author      Egor Izmaylov
#   * @brief       Runs the NPS/C-backend side of numerical verifier plans.
#   * @details     2026.06.28  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nn import Tensor

from .dtype import from_float32
from .runner_params import onnx_dtype_id_from_name


@dataclass(frozen=True)
class NpsForwardResult:
    output: object
    topk_indices: object | None = None


def _operator_init_args(op_name: str, init_args: dict, out_dtype: str) -> tuple[dict, dict[str, int]]:
    op_init_args = dict(init_args)
    op_init_args.pop("sizes_value", None)
    op_init_args.pop("size_value", None)
    op_init_args.pop("k_value", None)
    op_init_args.pop("dft_length_value", None)
    op_init_args.pop("frame_step_value", None)
    op_init_args.pop("frame_length_value", None)
    op_init_args.pop("dft_variant", None)
    op_init_args.pop("stft_variant", None)
    op_init_args.pop("target_shape", None)
    op_init_args.pop("repeats_value", None)
    op_init_args.pop("pads_value", None)
    op_init_args.pop("constant_value", None)
    op_init_args.pop("starts_value", None)
    op_init_args.pop("ends_value", None)
    op_init_args.pop("axes_value", None)
    op_init_args.pop("steps_value", None)
    op_init_args.pop("condition_value", None)
    op_init_args.pop("write_indices_value", None)
    op_init_args.pop("position_ids_value", None)
    op_init_args.pop("image_shape_value", None)
    op_init_args.pop("block_shape_value", None)
    op_init_args.pop("window_size_value", None)
    op_init_args.pop("start_value", None)
    op_init_args.pop("limit_value", None)
    op_init_args.pop("delta_value", None)
    op_init_args.pop("depth_value", None)
    op_init_args.pop("values_value", None)
    op_init_args.pop("sequence_lens_value", None)
    op_init_args.pop("num_mel_bins_value", None)
    op_init_args.pop("sample_rate_value", None)
    op_init_args.pop("lower_edge_hertz_value", None)
    op_init_args.pop("upper_edge_hertz_value", None)
    op_init_args.pop("input_values", None)
    op_init_args.pop("scale_values", None)
    op_init_args.pop("zero_point_values", None)
    controls = {
        "omit_zero_point": int(op_init_args.pop("omit_zero_point", 0)),
        "emit_log_prob": int(op_init_args.pop("emit_log_prob", 0)),
        "emit_stats": int(op_init_args.pop("emit_stats", 0)),
        "num_outputs": int(op_init_args.pop("num_outputs", len(init_args.get("split_value", [])) or 1)),
    }
    op_init_args.pop("ratio_value", None)
    op_init_args.pop("training_mode_value", None)
    op_init_args.pop("prob_values", None)
    op_init_args.pop("grid_variant", None)
    op_init_args.pop("roi_variant", None)
    op_init_args.pop("attention_mask_variant", None)
    op_init_args.pop("target_values", None)
    op_init_args.pop("weight_values", None)
    op_init_args.pop("score_values", None)
    op_init_args.pop("boxes_values", None)
    op_init_args.pop("scores_values", None)
    op_init_args.pop("max_output_value", None)
    op_init_args.pop("iou_threshold_value", None)
    op_init_args.pop("score_threshold_value", None)
    op_init_args.pop("split_value", None)
    op_init_args.pop("shape_value", None)
    fill_value = op_init_args.pop("fill_value", None)
    if op_name == "constant_of_shape" and fill_value is not None:
        op_init_args["value"] = from_float32(np.array([fill_value], dtype=np.float32), out_dtype)
    return op_init_args, controls


def _normalize_reduce_output(op_name: str, nps_out):
    if op_name not in {
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_l1",
        "reduce_l2",
        "reduce_log_sum",
        "reduce_log_sum_exp",
        "reduce_sum_square",
    }:
        return nps_out
    if np.shape(nps_out) == ():
        return np.array([float(nps_out)], dtype=np.float32)
    return np.asarray(nps_out, dtype=np.float32).reshape(1,)


def run_nps_forward(op_cls, op_name: str, inputs_tensor: list, init_args: dict, out_dtype: str) -> NpsForwardResult:
    op_init_args, controls = _operator_init_args(op_name, init_args, out_dtype)
    valid_tensors = [tensor for tensor in inputs_tensor if tensor is not None]
    topk_indices = None

    if op_name in {"random_uniform", "random_normal"}:
        op = op_cls(inputs=[], outputs=[], **op_init_args)
        nps_out = op.forward()["tensor"].data

    elif op_name in {"random_uniform_like", "random_normal_like"}:
        op = op_cls(inputs=[], outputs=[], **op_init_args)
        nps_out = op.forward(valid_tensors[0])["tensor"].data

    elif op_name == "dropout":
        op = op_cls(inputs=[], outputs=["y", "mask"], **op_init_args)
        nps_out = [tensor.data for tensor in op.forward(*valid_tensors)["tensor"]]

    elif op_name in {"conv2d", "conv_transpose", "gemm"}:
        op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **op_init_args)
        nps_out = op.forward(inputs_tensor[0], inputs_tensor[1], inputs_tensor[2])["tensor"].data

    elif op_name == "conv_integer":
        op = op_cls(inputs=[], outputs=[], **op_init_args)
        nps_out = op.forward(inputs_tensor[0], inputs_tensor[1], inputs_tensor[2], inputs_tensor[3])["tensor"].data

    elif op_name in {"rnn", "gru", "lstm"}:
        outputs = ["y", "y_h", "y_c"] if op_name == "lstm" else ["y", "y_h"]
        op = op_cls(inputs=[], outputs=outputs, dtype=out_dtype, **op_init_args)
        nps_out = [tensor.data for tensor in op.forward(*valid_tensors)["tensor"]]

    elif op_name == "batch_normalization":
        outputs = ["y", "running_mean", "running_var"] if int(op_init_args.get("training_mode", 0)) else ["y"]
        op = op_cls(inputs=[], outputs=outputs, dtype=out_dtype, **op_init_args)
        out = op.forward(*valid_tensors)["tensor"]
        nps_out = [tensor.data for tensor in out] if len(outputs) > 1 else out.data

    elif op_name == "layer_normalization" and controls["emit_stats"]:
        op = op_cls(inputs=[], outputs=["y", "mean", "inv_std"], dtype=out_dtype, **op_init_args)
        nps_out = [tensor.data for tensor in op.forward(*valid_tensors)["tensor"]]

    elif op_name in {"hann_window", "hamming_window", "blackman_window"}:
        op = op_cls(inputs=[], outputs=[], output_datatype=onnx_dtype_id_from_name(out_dtype), **op_init_args)
        nps_out = op.forward(valid_tensors[0])["tensor"].data

    elif op_name == "mel_weight_matrix":
        op = op_cls(inputs=[], outputs=[], output_datatype=onnx_dtype_id_from_name(out_dtype), **op_init_args)
        nps_out = op.forward(*valid_tensors)["tensor"].data

    elif op_name in {"tril", "triu", "trilu"}:
        op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **op_init_args)
        nps_out = op.forward(*valid_tensors)["tensor"].data

    elif op_name == "dynamic_quantize_linear":
        op = op_cls(inputs=[], outputs=["y", "y_scale", "y_zero_point"], **op_init_args)
        nps_out = [tensor.data for tensor in op.forward(valid_tensors[0])["tensor"]]

    elif op_name in {"quantize_linear", "dequantize_linear"} and controls["omit_zero_point"]:
        op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **op_init_args)
        nps_out = op.forward(inputs_tensor[0], inputs_tensor[1])["tensor"].data

    elif op_name == "split":
        outputs = [f"y{idx}" for idx in range(controls["num_outputs"])]
        op = op_cls(inputs=[], outputs=outputs, dtype=out_dtype, **op_init_args)
        nps_out = [tensor.data for tensor in op.forward(*valid_tensors)["tensor"]]

    elif op_name == "unique":
        op = op_cls(inputs=[], outputs=["y", "indices", "inverse", "counts"], dtype=out_dtype, **op_init_args)
        nps_out = [tensor.data for tensor in op.forward(valid_tensors[0])["tensor"]]

    elif op_name == "softmax_cross_entropy_loss":
        outputs = ["loss", "log_prob"] if controls["emit_log_prob"] else ["loss"]
        op = op_cls(inputs=[], outputs=outputs, dtype=out_dtype, **op_init_args)
        out = op.forward(*valid_tensors)["tensor"]
        nps_out = [tensor.data for tensor in out] if controls["emit_log_prob"] else out.data

    elif op_name == "negative_log_likelihood_loss":
        op = op_cls(inputs=[], outputs=["loss"], dtype=out_dtype, **op_init_args)
        nps_out = op.forward(*valid_tensors)["tensor"].data

    elif op_name == "stft":
        op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **op_init_args)
        nps_out = op.forward(inputs_tensor[0], inputs_tensor[1], inputs_tensor[2], inputs_tensor[3])["tensor"].data

    else:
        op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **op_init_args)

        if op_name in {"cumsum", "cumprod"}:
            axis_np = np.array([0], dtype=np.int64)
            axis_tensor = Tensor(*axis_np.shape, dtype="int64", data=axis_np)
            nps_out = op.forward(valid_tensors[0], axis_tensor)["tensor"].data

        elif op_name == "resize":
            nps_out = op.forward(valid_tensors[0], valid_tensors[1], valid_tensors[2], valid_tensors[3])["tensor"].data

        elif op_name == "topk":
            topk_ret = op.forward(valid_tensors[0], valid_tensors[1])["tensor"]
            nps_out = topk_ret[0].data
            topk_indices = topk_ret[1].data

        elif op_name == "cast_like":
            op = op_cls(inputs=[], outputs=[])
            nps_out = op.forward(valid_tensors[0], valid_tensors[1])["tensor"].data

        else:
            nps_out = op.forward(*valid_tensors)["tensor"].data

    return NpsForwardResult(_normalize_reduce_output(op_name, nps_out), topk_indices)
