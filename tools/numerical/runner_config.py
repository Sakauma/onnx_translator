# /**
#   ******************************************************************************
#   * @file        runner_config.py
#   * @author      Egor Izmaylov
#   * @brief       集中声明数值验证的容差策略和按算子族划分的 CUDA 执行属性。
#   * @details     2026.07.15  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from dataclasses import dataclass


OPERATOR_FAMILIES = {
    "complex_kernel": frozenset({
        "conv2d", "conv_integer", "qlinear_conv", "conv_transpose", "col2im",
        "deform_conv", "attention", "matmul_integer", "qlinear_matmul", "max_pool",
        "average_pool", "lp_pool", "global_average_pool", "global_max_pool",
        "global_lp_pool", "lrn", "mean_variance_normalization", "batch_normalization",
        "instance_normalization", "layer_normalization", "lp_normalization",
        "group_normalization", "negative_log_likelihood_loss",
        "softmax_cross_entropy_loss", "non_max_suppression", "max_unpool", "grid_sample",
        "max_roi_pool", "roi_align", "dft", "stft", "rnn", "gru", "lstm", "gemm",
        "softmax", "hardmax", "log_softmax",
    }),
    "double_kernel": frozenset({"quantize_linear", "dequantize_linear"}),
    "int64_passthrough": frozenset({
        "gather", "scatternd", "tensor_scatter", "scatter_elements", "gather_elements",
        "gathernd", "resize", "affine_grid", "topk", "max_unpool", "roi_align", "col2im",
        "dft", "stft", "rnn", "gru", "lstm", "tile", "expand", "pad",
        "center_crop_pad", "slice", "constant_of_shape", "rotary_embedding", "tril", "triu",
        "trilu", "hann_window", "hamming_window", "blackman_window", "range", "one_hot",
        "reverse_sequence", "mel_weight_matrix", "negative_log_likelihood_loss",
        "softmax_cross_entropy_loss", "non_max_suppression",
    }),
    "no_broadcast": frozenset({
        "matmul", "reduce_mean", "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
        "reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square",
        "gather", "gather_elements", "gathernd", "scatternd", "tensor_scatter",
        "scatter_elements", "nonzero", "argmin", "argmax", "size", "resize", "affine_grid",
        "grid_sample", "einsum", "topk", "random_uniform", "random_uniform_like",
        "random_normal", "random_normal_like", "bernoulli", "multinomial", "expand", "flatten",
        "reshape", "squeeze", "unsqueeze", "transpose", "tile", "concat", "pad",
        "center_crop_pad", "depth_to_space", "space_to_depth", "slice", "compress",
        "constant_of_shape", "eye_like", "rotary_embedding", "col2im", "deform_conv",
        "attention", "tril", "triu", "trilu", "hann_window", "hamming_window",
        "blackman_window", "range", "one_hot", "reverse_sequence", "det", "mel_weight_matrix",
        "negative_log_likelihood_loss", "softmax_cross_entropy_loss", "non_max_suppression",
        "quantize_linear", "dequantize_linear",
    }),
}


@dataclass(frozen=True)
class VerificationConfig:
    atol: float
    rtol: float
    complex_kernel: bool
    double_kernel: bool
    int64_passthrough: bool
    broadcast_inputs: bool


def resolve_verification_config(op_name, out_dtype):
    atol, rtol = 1e-4, 1e-4
    if "float16" in out_dtype:
        atol, rtol = 0.01, 0.01
    if "bfloat16" in out_dtype:
        atol, rtol = 0.1, 0.02
    if "float8" in out_dtype:
        atol, rtol = 0.1, 0.1
    if "int" in out_dtype:
        atol, rtol = 0.0, 0.0
    if op_name == "cos":
        atol = max(atol, 0.02)
    if op_name == "einsum":
        atol, rtol = max(atol, 1e-2), max(rtol, 1e-3)

    complex_kernel = op_name in OPERATOR_FAMILIES["complex_kernel"]
    return VerificationConfig(
        atol=atol,
        rtol=rtol,
        complex_kernel=complex_kernel,
        double_kernel=complex_kernel or op_name in OPERATOR_FAMILIES["double_kernel"],
        int64_passthrough=op_name in OPERATOR_FAMILIES["int64_passthrough"],
        broadcast_inputs=op_name not in OPERATOR_FAMILIES["no_broadcast"],
    )
