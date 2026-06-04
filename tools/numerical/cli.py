# /**
#   ******************************************************************************
#   * @file        cli.py
#   * @author      Egor Izmaylov
#   * @brief       提供 numerical_correctness 命令行入口和默认算子验证计划。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import nn
from nn.Operators import (
    Gemm, MaxPool, AveragePool, LpPool, GlobalAveragePool, GlobalMaxPool, GlobalLpPool,
    MaxRoiPool, RoiAlign,
    ADD, SUB, MUL, DIV, MatMul, MatMulInteger, QLinearMatMul,
    ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd,
    RELU, ABS, Pow, SQRT, Conv, ConvTranspose, ConvInteger, QLinearConv, ScatterND, Clip,
    Equal, Greater, Less, GreaterOrEqual, LessOrEqual,
    Gather, GatherElements, GatherND, COS, LOG, EXP, SIGMOID, TANH,
    Sin, Floor, Atan, Sign, Tan, Neg, Mod, Max, Min, Not, And, Or, Xor, IsNaN,
    CumSum, Softmax, NonZero, TopK, ArgMin, ArgMax, Resize, RandomUniformLike, Einsum,
    QuantizeLinear, DequantizeLinear, MaxUnpool, DFT, STFT, RNN, GRU, LSTM,
    Flatten, Reshape, Transpose, Tile, Concat, Expand, Pad, ConstantOfShape, EyeLike,
    Mean, Sum, Cast, CastLike, Ceil, Reciprocal, Softplus, Softsign, HardSigmoid,
    Elu, LeakyRelu, PRelu, Selu, Celu, ThresholdedRelu,
    HardSwish, Shrink, Gelu, Mish,
)

from . import cuda as cuda_backend
from .cuda import CUDA_VERIFY_DIR
from .runner import verify_op


def build_mixed_precision_plans():
    return [
        # ---- 混合精度基础算术和激活 ----
        (ADD, "add", [(32, 32), (32, 32)], ["float16", "float16"], "float16"),
        (ADD, "add", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bfloat16"),
        (ADD, "add", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (ADD, "add", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (SUB, "sub", [(32, 32), (32, 32)], ["float16", "float16"], "float16"),
        (SUB, "sub", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bfloat16"),
        (SUB, "sub", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (SUB, "sub", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (MUL, "mul", [(32, 32), (32, 32)], ["float16", "float16"], "float16"),
        (MUL, "mul", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bfloat16"),
        (MUL, "mul", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (MUL, "mul", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (DIV, "div", [(32, 32), (32, 32)], ["float16", "float16"], "float16"),
        (DIV, "div", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bfloat16"),
        (DIV, "div", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (DIV, "div", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (RELU, "relu", [(32, 32)], ["float16"], "float16"),
        (RELU, "relu", [(32, 32)], ["bfloat16"], "bfloat16"),
        (RELU, "relu", [(32, 32)], ["float8_e4m3"], "float8_e4m3"),
        (RELU, "relu", [(32, 32)], ["float8_e5m2"], "float8_e5m2"),
        (ABS, "abs", [(32, 32)], ["float16"], "float16"),
        (ABS, "abs", [(32, 32)], ["bfloat16"], "bfloat16"),
        (SIGMOID, "sigmoid", [(32, 32)], ["float16"], "float16"),
        (SIGMOID, "sigmoid", [(32, 32)], ["bfloat16"], "bfloat16"),
        (TANH, "tanh", [(32, 32)], ["float16"], "float16"),
        (TANH, "tanh", [(32, 32)], ["bfloat16"], "bfloat16"),
        (SQRT, "sqrt", [(32, 32)], ["float16"], "float16"),
        (SQRT, "sqrt", [(32, 32)], ["bfloat16"], "bfloat16"),
        (Max, "max", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bfloat16"),
        (Min, "min", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bfloat16"),
        (Max, "max", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (Min, "min", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (Max, "max", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (Min, "min", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (Clip, "clip", [(32, 32), (1,), (1,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16"),
        (Equal, "equal", [(32, 32), (32, 32)], ["float16", "float16"], "bool"),
        (Equal, "equal", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bool"),
        (Equal, "equal", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "bool"),
        (Equal, "equal", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "bool"),
        (Greater, "greater", [(32, 32), (32, 32)], ["float16", "float16"], "bool"),
        (Greater, "greater", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bool"),
        (Greater, "greater", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "bool"),
        (Greater, "greater", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "bool"),
        (Less, "less", [(32, 32), (32, 32)], ["float16", "float16"], "bool"),
        (Less, "less", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bool"),
        (Less, "less", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "bool"),
        (Less, "less", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "bool"),
        (GreaterOrEqual, "greater_or_equal", [(32, 32), (32, 32)], ["float16", "float16"], "bool"),
        (GreaterOrEqual, "greater_or_equal", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bool"),
        (GreaterOrEqual, "greater_or_equal", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "bool"),
        (GreaterOrEqual, "greater_or_equal", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "bool"),
        (LessOrEqual, "less_or_equal", [(32, 32), (32, 32)], ["float16", "float16"], "bool"),
        (LessOrEqual, "less_or_equal", [(32, 32), (32, 32)], ["bfloat16", "bfloat16"], "bool"),
        (LessOrEqual, "less_or_equal", [(32, 32), (32, 32)], ["float8_e4m3", "float8_e4m3"], "bool"),
        (LessOrEqual, "less_or_equal", [(32, 32), (32, 32)], ["float8_e5m2", "float8_e5m2"], "bool"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float16", "float16", "float16"], "float16"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e4m3", "float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e5m2", "float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float16", "float16", "float16"], "float16"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e4m3", "float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e5m2", "float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (Cast, "cast", [(8, 8)], ["float32"], "float16"),
        (Cast, "cast", [(8, 8)], ["float32"], "bfloat16"),
        (Cast, "cast", [(8, 8)], ["float32"], "float8_e4m3"),
        (Cast, "cast", [(8, 8)], ["float32"], "float8_e5m2"),
        (Cast, "cast", [(8, 8)], ["float16"], "float32"),
        (Cast, "cast", [(8, 8)], ["bfloat16"], "float32"),
        (Cast, "cast", [(8, 8)], ["float8_e4m3"], "float32"),
        (Cast, "cast", [(8, 8)], ["float8_e5m2"], "float32"),
        (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "float16"], "float16"),
        (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "bfloat16"], "bfloat16"),
        (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "float8_e4m3"], "float8_e4m3"),
        (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "float8_e5m2"], "float8_e5m2"),
        (Ceil, "ceil", [(16, 16)], ["float16"], "float16"),
        (Ceil, "ceil", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Ceil, "ceil", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Ceil, "ceil", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Reciprocal, "reciprocal", [(16, 16)], ["float16"], "float16"),
        (Reciprocal, "reciprocal", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Reciprocal, "reciprocal", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Reciprocal, "reciprocal", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Softplus, "softplus", [(16, 16)], ["float16"], "float16"),
        (Softplus, "softplus", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Softplus, "softplus", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Softplus, "softplus", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Softsign, "softsign", [(16, 16)], ["float16"], "float16"),
        (Softsign, "softsign", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Softsign, "softsign", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Softsign, "softsign", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (HardSigmoid, "hard_sigmoid", [(16, 16)], ["float16"], "float16", {"alpha": 0.2, "beta": 0.5}),
        (HardSigmoid, "hard_sigmoid", [(16, 16)], ["bfloat16"], "bfloat16", {"alpha": 0.2, "beta": 0.5}),
        (HardSigmoid, "hard_sigmoid", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"alpha": 0.2, "beta": 0.5}),
        (HardSigmoid, "hard_sigmoid", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"alpha": 0.2, "beta": 0.5}),
        (Elu, "elu", [(16, 16)], ["float16"], "float16", {"alpha": 0.7}),
        (Elu, "elu", [(16, 16)], ["bfloat16"], "bfloat16", {"alpha": 0.7}),
        (Elu, "elu", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"alpha": 0.7}),
        (Elu, "elu", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"alpha": 0.7}),
        (LeakyRelu, "leaky_relu", [(16, 16)], ["float16"], "float16", {"alpha": 0.25}),
        (LeakyRelu, "leaky_relu", [(16, 16)], ["bfloat16"], "bfloat16", {"alpha": 0.25}),
        (LeakyRelu, "leaky_relu", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"alpha": 0.25}),
        (LeakyRelu, "leaky_relu", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"alpha": 0.25}),
        (PRelu, "prelu", [(2, 3, 4), (1, 3, 1)], ["float16", "float16"], "float16"),
        (PRelu, "prelu", [(2, 3, 4), (1, 3, 1)], ["bfloat16", "bfloat16"], "bfloat16"),
        (PRelu, "prelu", [(2, 3, 4), (1, 3, 1)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (PRelu, "prelu", [(2, 3, 4), (1, 3, 1)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (Selu, "selu", [(16, 16)], ["float16"], "float16", {"alpha": 1.67326, "gamma": 1.0507}),
        (Selu, "selu", [(16, 16)], ["bfloat16"], "bfloat16", {"alpha": 1.67326, "gamma": 1.0507}),
        (Selu, "selu", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"alpha": 1.67326, "gamma": 1.0507}),
        (Selu, "selu", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"alpha": 1.67326, "gamma": 1.0507}),
        (Celu, "celu", [(16, 16)], ["float16"], "float16", {"alpha": 0.7}),
        (Celu, "celu", [(16, 16)], ["bfloat16"], "bfloat16", {"alpha": 0.7}),
        (Celu, "celu", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"alpha": 0.7}),
        (Celu, "celu", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"alpha": 0.7}),
        (ThresholdedRelu, "thresholded_relu", [(16, 16)], ["float16"], "float16", {"alpha": 0.3}),
        (ThresholdedRelu, "thresholded_relu", [(16, 16)], ["bfloat16"], "bfloat16", {"alpha": 0.3}),
        (ThresholdedRelu, "thresholded_relu", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"alpha": 0.3}),
        (ThresholdedRelu, "thresholded_relu", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"alpha": 0.3}),
        (HardSwish, "hard_swish", [(16, 16)], ["float16"], "float16"),
        (HardSwish, "hard_swish", [(16, 16)], ["bfloat16"], "bfloat16"),
        (HardSwish, "hard_swish", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (HardSwish, "hard_swish", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Shrink, "shrink", [(16, 16)], ["float16"], "float16", {"bias": 0.2, "lambd": 0.5}),
        (Shrink, "shrink", [(16, 16)], ["bfloat16"], "bfloat16", {"bias": 0.2, "lambd": 0.5}),
        (Shrink, "shrink", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"bias": 0.2, "lambd": 0.5}),
        (Shrink, "shrink", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"bias": 0.2, "lambd": 0.5}),
        (Gelu, "gelu", [(16, 16)], ["float16"], "float16"),
        (Gelu, "gelu", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Gelu, "gelu", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Gelu, "gelu", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Gelu, "gelu", [(16, 16)], ["float16"], "float16", {"approximate": "tanh"}),
        (Gelu, "gelu", [(16, 16)], ["bfloat16"], "bfloat16", {"approximate": "tanh"}),
        (Gelu, "gelu", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"approximate": "tanh"}),
        (Gelu, "gelu", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"approximate": "tanh"}),
        (Mish, "mish", [(16, 16)], ["float16"], "float16"),
        (Mish, "mish", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Mish, "mish", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Mish, "mish", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),

        # ---- 混合精度矩阵、卷积、池化和 ROI ----
        (MatMul, "matmul", [(16, 32), (32, 8)], ["float16", "float16"], "float16"),
        (MatMul, "matmul", [(16, 32), (32, 8)], ["bfloat16", "bfloat16"], "bfloat16"),
        (Gemm, "gemm", [(8, 16), (16, 4), (4,)], ["float16", "float16", "float16"], "float16", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}),
        (Gemm, "gemm", [(8, 16), (16, 4), (4,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}),
        (Conv, "conv2d", [(1, 1, 5, 5), (1, 1, 3, 3), (1,)], ["float16", "float16", "float16"], "float16", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1}),
        (Conv, "conv2d", [(1, 1, 5, 5), (1, 1, 3, 3), (1,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1}),
        (ConvTranspose, "conv_transpose", [(1, 2, 4, 4), (2, 3, 3, 3), (3,)], ["float16", "float16", "float16"], "float16", {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "group": 1, "output_padding": [1, 1]}),
        (ConvTranspose, "conv_transpose", [(1, 2, 4, 4), (2, 3, 3, 3), (3,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "group": 1, "output_padding": [1, 1]}),
        (MaxPool, "max_pool", [(1, 2, 8, 8)], ["float16"], "float16", {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]}),
        (MaxPool, "max_pool", [(1, 2, 8, 8)], ["bfloat16"], "bfloat16", {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]}),
        (AveragePool, "average_pool", [(1, 2, 7, 7)], ["float16"], "float16", {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "count_include_pad": 1}),
        (AveragePool, "average_pool", [(1, 2, 7, 7)], ["bfloat16"], "bfloat16", {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "count_include_pad": 1}),
        (GlobalAveragePool, "global_average_pool", [(1, 3, 5, 4)], ["float16"], "float16"),
        (GlobalAveragePool, "global_average_pool", [(1, 3, 5, 4)], ["bfloat16"], "bfloat16"),
        (MaxRoiPool, "max_roi_pool", [(2, 2, 5, 5), (2, 5)], ["float16", "float16"], "float16", {"pooled_shape": [2, 3], "spatial_scale": 1.0}),
        (MaxRoiPool, "max_roi_pool", [(2, 2, 5, 5), (2, 5)], ["bfloat16", "bfloat16"], "bfloat16", {"pooled_shape": [2, 3], "spatial_scale": 1.0}),
        (RoiAlign, "roi_align", [(2, 1, 4, 5), (2, 4), (2,)], ["float16", "float16", "int64"], "float16", {"output_height": 2, "output_width": 3, "sampling_ratio": 2, "spatial_scale": 1.0, "mode": "avg", "coordinate_transformation_mode": "half_pixel"}),
        (RoiAlign, "roi_align", [(2, 1, 4, 5), (2, 4), (2,)], ["bfloat16", "bfloat16", "int64"], "bfloat16", {"output_height": 2, "output_width": 3, "sampling_ratio": 2, "spatial_scale": 1.0, "mode": "avg", "coordinate_transformation_mode": "half_pixel"}),

        # ---- 混合精度归约、排序、量化、谱和循环网络 ----
        (ReduceSum, "reduce_sum", [(32, 32)], ["float16"], "float16", {"axes": None, "keepdims": 0}),
        (ReduceSum, "reduce_sum", [(32, 32)], ["bfloat16"], "bfloat16", {"axes": None, "keepdims": 0}),
        (ReduceMean, "reduce_mean", [(32, 32)], ["float16"], "float16"),
        (ReduceMean, "reduce_mean", [(32, 32)], ["bfloat16"], "bfloat16"),
        (Softmax, "softmax", [(4, 16)], ["float16"], "float16", {"axis": -1}),
        (Softmax, "softmax", [(4, 16)], ["bfloat16"], "bfloat16", {"axis": -1}),
        (TopK, "topk", [(16, 16), (1,)], ["float16", "int64"], "float16", {"axis": 1, "largest": 1, "sorted": 1, "k_value": 4}),
        (TopK, "topk", [(16, 16), (1,)], ["bfloat16", "int64"], "bfloat16", {"axis": 1, "largest": 1, "sorted": 1, "k_value": 4}),
        (ArgMin, "argmin", [(16, 16)], ["float16"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (ArgMin, "argmin", [(16, 16)], ["bfloat16"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (ArgMin, "argmin", [(16, 16)], ["float8_e4m3"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (ArgMin, "argmin", [(16, 16)], ["float8_e5m2"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (ArgMax, "argmax", [(16, 16)], ["float16"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (ArgMax, "argmax", [(16, 16)], ["bfloat16"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (ArgMax, "argmax", [(16, 16)], ["float8_e4m3"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (ArgMax, "argmax", [(16, 16)], ["float8_e5m2"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),
        (Gather, "gather", [(16, 16), (4,)], ["float16", "int64"], "float16", {"axis": 0}),
        (Gather, "gather", [(16, 16), (4,)], ["bfloat16", "int64"], "bfloat16", {"axis": 0}),
        (Gather, "gather", [(16, 16), (4,)], ["float8_e4m3", "int64"], "float8_e4m3", {"axis": 0}),
        (Gather, "gather", [(16, 16), (4,)], ["float8_e5m2", "int64"], "float8_e5m2", {"axis": 0}),
        (GatherElements, "gather_elements", [(16, 16), (16, 16)], ["float16", "int64"], "float16", {"axis": 1}),
        (GatherElements, "gather_elements", [(16, 16), (16, 16)], ["bfloat16", "int64"], "bfloat16", {"axis": 1}),
        (GatherElements, "gather_elements", [(16, 16), (16, 16)], ["float8_e4m3", "int64"], "float8_e4m3", {"axis": 1}),
        (GatherElements, "gather_elements", [(16, 16), (16, 16)], ["float8_e5m2", "int64"], "float8_e5m2", {"axis": 1}),
        (GatherND, "gathernd", [(16, 16), (64, 2)], ["float16", "int64"], "float16"),
        (GatherND, "gathernd", [(16, 16), (64, 2)], ["bfloat16", "int64"], "bfloat16"),
        (GatherND, "gathernd", [(16, 16), (64, 2)], ["float8_e4m3", "int64"], "float8_e4m3"),
        (GatherND, "gathernd", [(16, 16), (64, 2)], ["float8_e5m2", "int64"], "float8_e5m2"),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["float16", "int64", "float16"], "float16"),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["bfloat16", "int64", "bfloat16"], "bfloat16"),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["float8_e4m3", "int64", "float8_e4m3"], "float8_e4m3"),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["float8_e5m2", "int64", "float8_e5m2"], "float8_e5m2"),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["float16", "float16", "float16", "int64"], "float16", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["bfloat16", "bfloat16", "bfloat16", "int64"], "bfloat16", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["float8_e4m3", "float8_e4m3", "float8_e4m3", "int64"], "float8_e4m3", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["float8_e5m2", "float8_e5m2", "float8_e5m2", "int64"], "float8_e5m2", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (Expand, "expand", [(2, 1, 3), (3,)], ["float16", "int64"], "float16", {"target_shape": [2, 4, 3]}),
        (Expand, "expand", [(2, 1, 3), (3,)], ["bfloat16", "int64"], "bfloat16", {"target_shape": [2, 4, 3]}),
        (Expand, "expand", [(2, 1, 3), (3,)], ["float8_e4m3", "int64"], "float8_e4m3", {"target_shape": [2, 4, 3]}),
        (Expand, "expand", [(2, 1, 3), (3,)], ["float8_e5m2", "int64"], "float8_e5m2", {"target_shape": [2, 4, 3]}),
        (ConstantOfShape, "constant_of_shape", [(3,)], ["int64"], "float16", {"shape_value": [2, 3, 4], "fill_value": -1.5}),
        (ConstantOfShape, "constant_of_shape", [(3,)], ["int64"], "bfloat16", {"shape_value": [2, 3, 4], "fill_value": -1.5}),
        (ConstantOfShape, "constant_of_shape", [(3,)], ["int64"], "float8_e4m3", {"shape_value": [2, 3, 4], "fill_value": -1.5}),
        (ConstantOfShape, "constant_of_shape", [(3,)], ["int64"], "float8_e5m2", {"shape_value": [2, 3, 4], "fill_value": -1.5}),
        (EyeLike, "eye_like", [(4, 5)], ["float16"], "float16", {"k": 1}),
        (EyeLike, "eye_like", [(4, 5)], ["bfloat16"], "bfloat16", {"k": 1}),
        (EyeLike, "eye_like", [(4, 5)], ["float8_e4m3"], "float8_e4m3", {"k": 1}),
        (EyeLike, "eye_like", [(4, 5)], ["float8_e5m2"], "float8_e5m2", {"k": 1}),
        (Flatten, "flatten", [(2, 3, 4)], ["float16"], "float16", {"axis": -1}),
        (Flatten, "flatten", [(2, 3, 4)], ["bfloat16"], "bfloat16", {"axis": -1}),
        (Flatten, "flatten", [(2, 3, 4)], ["float8_e4m3"], "float8_e4m3", {"axis": -1}),
        (Flatten, "flatten", [(2, 3, 4)], ["float8_e5m2"], "float8_e5m2", {"axis": -1}),
        (Reshape, "reshape", [(2, 3, 4), (2,)], ["float16", "int64"], "float16", {"target_shape": [0, -1]}),
        (Reshape, "reshape", [(2, 3, 4), (2,)], ["bfloat16", "int64"], "bfloat16", {"target_shape": [0, -1]}),
        (Reshape, "reshape", [(2, 3, 4), (2,)], ["float8_e4m3", "int64"], "float8_e4m3", {"target_shape": [0, -1]}),
        (Reshape, "reshape", [(2, 3, 4), (2,)], ["float8_e5m2", "int64"], "float8_e5m2", {"target_shape": [0, -1]}),
        (Transpose, "transpose", [(2, 3, 4)], ["float16"], "float16", {"perm": [2, 0, 1]}),
        (Transpose, "transpose", [(2, 3, 4)], ["bfloat16"], "bfloat16", {"perm": [2, 0, 1]}),
        (Transpose, "transpose", [(2, 3, 4)], ["float8_e4m3"], "float8_e4m3", {"perm": [2, 0, 1]}),
        (Transpose, "transpose", [(2, 3, 4)], ["float8_e5m2"], "float8_e5m2", {"perm": [2, 0, 1]}),
        (Tile, "tile", [(2, 3), (2,)], ["float16", "int64"], "float16", {"repeats_value": [2, 3]}),
        (Tile, "tile", [(2, 3), (2,)], ["bfloat16", "int64"], "bfloat16", {"repeats_value": [2, 3]}),
        (Tile, "tile", [(2, 3), (2,)], ["float8_e4m3", "int64"], "float8_e4m3", {"repeats_value": [2, 3]}),
        (Tile, "tile", [(2, 3), (2,)], ["float8_e5m2", "int64"], "float8_e5m2", {"repeats_value": [2, 3]}),
        (Concat, "concat", [(2, 2, 4), (2, 3, 4)], ["float16", "float16"], "float16", {"axis": 1}),
        (Concat, "concat", [(2, 2, 4), (2, 3, 4)], ["bfloat16", "bfloat16"], "bfloat16", {"axis": 1}),
        (Concat, "concat", [(2, 2, 4), (2, 3, 4)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3", {"axis": 1}),
        (Concat, "concat", [(2, 2, 4), (2, 3, 4)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2", {"axis": 1}),
        (Pad, "pad", [(2, 3, 4), (6,), (1,)], ["float16", "int64", "float16"], "float16", {"mode": "constant", "pads_value": [0, 1, 1, 0, 1, 0], "constant_value": -2.0}),
        (Pad, "pad", [(2, 3, 4), (6,), (1,)], ["bfloat16", "int64", "bfloat16"], "bfloat16", {"mode": "constant", "pads_value": [0, 1, 1, 0, 1, 0], "constant_value": -2.0}),
        (Pad, "pad", [(2, 3, 4), (6,), (1,)], ["float8_e4m3", "int64", "float8_e4m3"], "float8_e4m3", {"mode": "constant", "pads_value": [0, 1, 1, 0, 1, 0], "constant_value": -2.0}),
        (Pad, "pad", [(2, 3, 4), (6,), (1,)], ["float8_e5m2", "int64", "float8_e5m2"], "float8_e5m2", {"mode": "constant", "pads_value": [0, 1, 1, 0, 1, 0], "constant_value": -2.0}),
        (QuantizeLinear, "quantize_linear", [(32, 32), (1,), (1,)], ["float16", "float16", "int8"], "int8"),
        (QuantizeLinear, "quantize_linear", [(32, 32), (1,), (1,)], ["bfloat16", "bfloat16", "int8"], "int8"),
        (DequantizeLinear, "dequantize_linear", [(32, 32), (1,), (1,)], ["int8", "float16", "int8"], "float16"),
        (DequantizeLinear, "dequantize_linear", [(32, 32), (1,), (1,)], ["int8", "bfloat16", "int8"], "bfloat16"),
        (QLinearMatMul, "qlinear_matmul", [(4, 6), (4,), (4,), (6, 5), (5,), (5,), (1,), (1,)], ["uint8", "bfloat16", "uint8", "uint8", "bfloat16", "uint8", "bfloat16", "uint8"], "uint8"),
        (QLinearConv, "qlinear_conv", [(1, 2, 5, 5), (1,), (1,), (2, 2, 3, 3), (2,), (2,), (1,), (1,)], ["uint8", "bfloat16", "uint8", "uint8", "bfloat16", "uint8", "bfloat16", "uint8"], "uint8", {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "group": 1}),
        (DFT, "dft", [(1, 4, 1), ()], ["float16", "int64"], "float16", {"axis": 1, "onesided": 1, "inverse": 0, "dft_length_value": 4}),
        (DFT, "dft", [(1, 4, 1), ()], ["bfloat16", "int64"], "bfloat16", {"axis": 1, "onesided": 1, "inverse": 0, "dft_length_value": 4}),
        (STFT, "stft", [(1, 4, 1), (), (2,), ()], ["float16", "int64", "float16", "int64"], "float16", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 2}),
        (STFT, "stft", [(1, 4, 1), (), (2,), ()], ["bfloat16", "int64", "bfloat16", "int64"], "bfloat16", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 2}),
        (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0}),
        (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 1}),
        (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float16", "float16", "float16", "float16", "int64", "float16", "float16", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 1}),
    ]


def build_default_plans():
    return [
    # ---- 四则运算 ----
    (ADD, "add", [(64,64), (64,64)], ["float32", "float32"], "float32"),
    (SUB, "sub", [(64,64), (64,64)], ["float32", "float32"], "float32"),
    (MUL, "mul", [(64,64), (64,64)], ["float32", "float32"], "float32"),
    (DIV, "div", [(64,64), (64,64)], ["float32", "float32"], "float32"),

    # ---- 常见广播----
    (ADD, "add", [(10, 10, 10), (10, 1)], ["float32", "float32"], "float32"),
    (SUB, "sub", [(4, 1, 16), (16,)], ["float32", "float32"], "float32"),

    # ---- 激活 ----
    (RELU, "relu", [(128,128)], ["float32"], "float32"),
    (ABS, "abs", [(128,128)], ["float32"], "float32"),

    # ---- Conv ----
    (Conv, "conv2d",[(1, 1, 5, 5), (1, 1, 3, 3), (1,)],["float32", "float32", "float32"], "float32",{"pads":[0,0,0,0], "strides":[1,1], "dilations":[1,1], "group":1}),
    (ConvInteger, "conv_integer",[(1, 2, 5, 5), (2, 2, 3, 3), (1,), (2,)],["uint8", "int8", "uint8", "int8"], "int32",{"pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "group":1}),
    (QLinearConv, "qlinear_conv",[(1, 2, 5, 5), (1,), (1,), (2, 2, 3, 3), (2,), (2,), (1,), (1,)],["uint8", "float32", "uint8", "uint8", "float32", "uint8", "float32", "uint8"], "uint8",{"pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "group":1}),
    (ConvTranspose, "conv_transpose",[(1, 2, 4, 4), (2, 3, 3, 3), (3,)],["float32", "float32", "float32"], "float32",{"pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "group":1, "output_padding":[1,1]}),

    # ---- Softmax ----
    (Softmax, "softmax",[(4, 64)], ["float32"], "float32", {"axis":-1}),

    # ---- Gemm ----
    (Gemm, "gemm",[(16, 32), (32, 8), (8,)], ["float32", "float32", "float32"], "float32",{"alpha":1.0, "beta":1.0, "transA":0, "transB":0}),

    # ---- MaxPool ----
    (MaxPool, "max_pool",[(1, 2, 16, 16)], ["float32"], "float32",{"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),
    (AveragePool, "average_pool",[(1, 2, 7, 7)], ["float32"], "float32",{"kernel_shape":[3,3], "pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "count_include_pad":1}),
    (LpPool, "lp_pool",[(1, 2, 7, 7)], ["float32"], "float32",{"kernel_shape":[3,3], "pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "p":2}),
    (GlobalAveragePool, "global_average_pool",[(1, 3, 5, 4)], ["float32"], "float32"),
    (GlobalMaxPool, "global_max_pool",[(1, 3, 5, 4)], ["float32"], "float32"),
    (GlobalLpPool, "global_lp_pool",[(1, 3, 5, 4)], ["float32"], "float32", {"p": 2}),
    (MaxUnpool, "max_unpool",[(1, 1, 2, 2), (1, 1, 2, 2)], ["float32", "int64"], "float32",{"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),
    (MaxRoiPool, "max_roi_pool",[(2, 2, 5, 5), (2, 5)], ["float32", "float32"], "float32", {"pooled_shape":[2, 3], "spatial_scale":1.0}),
    (RoiAlign, "roi_align",[(2, 1, 4, 5), (2, 4), (2,)], ["float32", "float32", "int64"], "float32", {"output_height":2, "output_width":3, "sampling_ratio":2, "spatial_scale":1.0, "mode":"avg", "coordinate_transformation_mode":"half_pixel"}),

    (Equal,   "equal",   [(64,64), (64,64)], ["float32", "float32"], "bool"),
    (Greater, "greater", [(64,64), (64,64)], ["float32", "float32"], "bool"),
    (Less,    "less",    [(64,64), (64,64)], ["float32", "float32"], "bool"),

    (Clip, "clip",[(64,64), (1,), (1,)],["float32", "float32", "float32"],"float32"),

    (QuantizeLinear, "quantize_linear", [(64,64), (1,), (1,)], ["float32", "float32", "int8"], "int8"),
    (DequantizeLinear, "dequantize_linear", [(64,64), (1,), (1,)], ["int8", "float32", "int8"], "float32"),

    (SQRT, "sqrt", [(64, 64)], ["float32"], "float32"),

    (Pow, "pow", [(64,64), (64,64)], ["float32", "float32"], "float32"),

    (MatMul, "matmul",[(32, 64), (64,16)],["float32", "float32"],"float32"),
    (MatMulInteger, "matmul_integer", [(4, 6), (6, 5), (4,), (5,)], ["uint8", "int8", "uint8", "int8"], "int32"),
    (QLinearMatMul, "qlinear_matmul", [(4, 6), (4,), (4,), (6, 5), (5,), (5,), (1,), (1,)], ["uint8", "float32", "uint8", "uint8", "float32", "uint8", "float32", "uint8"], "uint8"),

    (ReduceMean, "reduce_mean",[(32, 64)],["float32"], "float32"),

    (Gather, "gather",[(32, 64), (8,)],["float32", "int64"],"float32",{"axis": 0}),

    (ScatterND, "scatternd",[(32, 64), (16, 2), (16,)],["float32", "int64", "float32"],"float32"), 

    (SIGMOID, "sigmoid", [(256,256)], ["float32"], "float32"),
    (COS, "cos", [(256,256)], ["float32"], "float32"),
    (Sin, "sin", [(256,256)], ["float32"], "float32"),
    (LOG, "log", [(256,256)], ["float32"], "float32"),
    (Floor, "floor", [(256,256)], ["float32"], "float32"),
    (EXP, "exp", [(256,256)], ["float32"], "float32"),
    (Atan, "atan", [(256,256)], ["float32"], "float32"),
    (Sign, "sign", [(256,256)], ["float32"], "float32"),
    (Tan, "tan", [(256,256)], ["float32"], "float32"),
    (TANH, "tanh", [(256,256)], ["float32"], "float32"),
    (Neg, "neg", [(256,256)], ["float32"], "float32"),
    (Mod, "mod", [(256,256), (256,256)], ["float32", "float32"], "float32"),
    (Max, "max", [(256,256), (256,256)], ["float32", "float32"], "float32"),
    (Min, "min", [(256,256), (256,256)], ["float32", "float32"], "float32"),
    (IsNaN, "isnan", [(256,256)], ["float32"], "bool"),

    # 归约（简化：2D 全归约）
    (ReduceSum, "reduce_sum", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceMax, "reduce_max", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceMin, "reduce_min", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceProd, "reduce_prod", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),

    # 逻辑（bool 输入/输出）
    (Not, "not", [(256,256)], ["bool"], "bool"),
    (And, "and", [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (Or,  "or",  [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (Xor, "xor", [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (GreaterOrEqual, "greater_or_equal", [(256,256), (256,256)], ["float32", "float32"], "bool"),
    (LessOrEqual, "less_or_equal", [(256,256), (256,256)], ["float32", "float32"], "bool"),
    (Mean, "mean", [(16, 1, 8), (1, 4, 8), (16, 4, 1)], ["float32", "float32", "float32"], "float32"),
    (Sum, "sum", [(16, 1, 8), (1, 4, 8), (16, 4, 1)], ["float32", "float32", "float32"], "float32"),
    (Cast, "cast", [(8, 8)], ["float32"], "int64"),
    (Cast, "cast", [(8, 8)], ["float32"], "bool"),
    (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "int64"], "int64"),
    (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "bool"], "bool"),
    (Ceil, "ceil", [(64, 64)], ["float32"], "float32"),
    (Reciprocal, "reciprocal", [(64, 64)], ["float32"], "float32"),
    (Softplus, "softplus", [(64, 64)], ["float32"], "float32"),
    (Softsign, "softsign", [(64, 64)], ["float32"], "float32"),
    (HardSigmoid, "hard_sigmoid", [(64, 64)], ["float32"], "float32", {"alpha": 0.2, "beta": 0.5}),
    (Elu, "elu", [(64, 64)], ["float32"], "float32", {"alpha": 0.7}),
    (LeakyRelu, "leaky_relu", [(64, 64)], ["float32"], "float32", {"alpha": 0.25}),
    (PRelu, "prelu", [(2, 3, 4), (1, 3, 1)], ["float32", "float32"], "float32"),
    (Selu, "selu", [(64, 64)], ["float32"], "float32", {"alpha": 1.67326, "gamma": 1.0507}),
    (Celu, "celu", [(64, 64)], ["float32"], "float32", {"alpha": 0.7}),
    (ThresholdedRelu, "thresholded_relu", [(64, 64)], ["float32"], "float32", {"alpha": 0.3}),
    (HardSwish, "hard_swish", [(64, 64)], ["float32"], "float32"),
    (Shrink, "shrink", [(64, 64)], ["float32"], "float32", {"bias": 0.2, "lambd": 0.5}),
    (Gelu, "gelu", [(64, 64)], ["float32"], "float32"),
    (Gelu, "gelu", [(64, 64)], ["float32"], "float32", {"approximate": "tanh"}),
    (Mish, "mish", [(64, 64)], ["float32"], "float32"),

    # 索引
    (GatherElements, "gather_elements", [(64,64), (64,64)], ["float32", "int64"], "float32", {"axis":1}),
    (GatherND, "gathernd", [(64,64), (256,2)], ["float32", "int64"], "float32"),

    # 扫描
    (CumSum, "cumsum", [(1024,)], ["float32"], "float32", {"exclusive":0, "reverse":0}),

    (NonZero, "nonzero", [(64,64)], ["float32"], "int64"),

    (ArgMin, "argmin", [(64,64)], ["float32"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),

    (ArgMax, "argmax", [(64,64)], ["float32"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),

    # Resize: x, roi, scales, sizes
    (Resize, "resize", [(1,3,8,8), (0,), (0,), (4,)], ["float32", "float32", "float32", "int64"], "float32", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1,3,16,16]}),

    (Expand, "expand", [(2, 1, 3), (3,)], ["float32", "int64"], "float32", {"target_shape": [2, 4, 3]}),
    (ConstantOfShape, "constant_of_shape", [(3,)], ["int64"], "float32", {"shape_value": [2, 3, 4], "fill_value": -1.5}),
    (EyeLike, "eye_like", [(4, 5)], ["float32"], "float32", {"k": 1}),
    (Flatten, "flatten", [(2, 3, 4)], ["float32"], "float32", {"axis": -1}),
    (Reshape, "reshape", [(2, 3, 4), (2,)], ["float32", "int64"], "float32", {"target_shape": [0, -1]}),
    (Transpose, "transpose", [(2, 3, 4)], ["float32"], "float32", {"perm": [2, 0, 1]}),
    (Tile, "tile", [(2, 3), (2,)], ["float32", "int64"], "float32", {"repeats_value": [2, 3]}),
    (Concat, "concat", [(2, 2, 4), (2, 3, 4)], ["float32", "float32"], "float32", {"axis": 1}),
    (Pad, "pad", [(2, 3, 4), (6,), (1,)], ["float32", "int64", "float32"], "float32", {"mode": "constant", "pads_value": [0, 1, 1, 0, 1, 0], "constant_value": -2.0}),

    # Einsum: 当前固定主路径 ij,jk->ik
    (Einsum, "einsum", [(16,32), (32,8)], ["float32", "float32"], "float32", {"equation": "ij,jk->ik"}),

    (TopK, "topk", [(32, 64), (1,)], ["float32", "int64"], "float32",{"axis": 1, "largest": 1, "sorted": 1, "k_value": 8}),

    (RandomUniformLike, "random_uniform_like", [(32, 32)], ["float32"], "float32", {"low": -1.0, "high": 1.0, "seed": 123}),

    (DFT, "dft", [(1, 4, 1), ()], ["float32", "int64"], "float32", {"axis": 1, "onesided": 1, "inverse": 0, "dft_length_value": 4}),
    (STFT, "stft", [(1, 4, 1), (), (2,), ()], ["float32", "int64", "float32", "int64"], "float32", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 2}),
    (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0}),
    (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 1}),
    (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float32", "float32", "float32", "float32", "int64", "float32", "float32", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 1}),
] + build_mixed_precision_plans()



def main(argv=None):
    parser = argparse.ArgumentParser(description="Run C backend vs CUDA reference numerical checks.")
    parser.add_argument("--iterations", type=int, default=20, help="Iterations per test plan.")
    parser.add_argument("--op", action="append", help="Run only the named op. Can be repeated.")
    parser.add_argument("--cuda-dir", default=CUDA_VERIFY_DIR, help="Directory containing verify_* CUDA executables.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip matplotlib histogram generation.")
    args = parser.parse_args(argv)
    cuda_backend.CUDA_VERIFY_DIR = args.cuda_dir

    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        print(f"ERROR: C backend library not found: {nn.TENSOR_OPS_LIB_PATH}")
        print("Run `make` before numerical verification.")
        sys.exit(2)

    plans = build_default_plans()

    if args.op:
        selected_ops = set(args.op)
        plans = [plan for plan in plans if len(plan) >= 2 and plan[1] in selected_ops]
        missing_ops = selected_ops - {plan[1] for plan in plans}
        if missing_ops:
            print(f"ERROR: requested op(s) not found in test plans: {sorted(missing_ops)}")
            sys.exit(2)

    if not plans:
        print("ERROR: no numerical test plans selected.")
        sys.exit(2)

    print("🚀 开始数值验证 ...")
    ops_stats = {}
    failed_ops = []
    for plan in plans:
        if len(plan) == 5:
            op_cls, op_name, shapes, dtypes, out_dtype = plan
            init_args = {}
        elif len(plan) == 6:
            op_cls, op_name, shapes, dtypes, out_dtype, init_args = plan
        else:
            print(f"⚠️ 跳过格式错误的测试计划: {plan}")
            failed_ops.append("<malformed-plan>")
            continue
        abs_errs, rel_errs, ok = verify_op(
            op_cls,
            op_name,
            shapes,
            dtypes,
            out_dtype,
            init_args=init_args,
            iterations=args.iterations,
        )
        if not ok:
            failed_ops.append(op_name)
        # 按算子名称聚合数据
        if op_name not in ops_stats:
            ops_stats[op_name] = {'abs': [], 'rel': []}
        ops_stats[op_name]['abs'].extend(abs_errs)
        ops_stats[op_name]['rel'].extend(rel_errs)
    if not args.skip_plots:
        print("\n📊 正在按算子绘制误差分布直方图...")
        for op_name, stats in ops_stats.items():
            valid_abs = [x for x in stats['abs'] if np.isfinite(x) and x >= 0]
            valid_rel = [x for x in stats['rel'] if np.isfinite(x) and x >= 0]

            if len(valid_abs) == 0 or len(valid_rel) == 0:
                print(f"⚠️ [{op_name.upper()}] 没有可用的有限误差数据，跳过绘图")
                continue

            plt.figure(figsize=(14, 6))
            # --- 子图 1: 绝对误差分布 ---
            plt.subplot(1, 2, 1)
            plt.hist(valid_abs, bins=50, color='skyblue', edgecolor='black', log=True)
            plt.title(f'Operator [{op_name.upper()}] - Absolute Error Dist')
            plt.xlabel('Max Absolute Error')
            plt.ylabel('Count (Log Scale)')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            # 标注 99% 分位数 (P99)
            if len(valid_abs) > 0:
                p99_abs = np.percentile(valid_abs, 99)
                plt.axvline(p99_abs, color='red', linestyle='dashed', linewidth=1)
                plt.text(p99_abs, plt.ylim()[1]*0.9, f' P99: {p99_abs:.2e}', color='red')
            # --- 子图 2: 相对误差分布 ---
            plt.subplot(1, 2, 2)
            plt.hist(valid_rel, bins=50, color='salmon', edgecolor='black', log=True)
            plt.title(f'Operator [{op_name.upper()}] - Relative Error Dist')
            plt.xlabel('Max Relative Error')
            plt.ylabel('Count (Log Scale)')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            # 标注 99% 分位数 (P99)
            if len(valid_rel) > 0:
                p99_rel = np.percentile(valid_rel, 99)
                plt.axvline(p99_rel, color='red', linestyle='dashed', linewidth=1)
                plt.text(p99_rel, plt.ylim()[1]*0.9, f' P99: {p99_rel:.2e}', color='red')

            plt.tight_layout()
            plt.close()
    print("\n📈 详细统计报告 (99th Percentile Summary):")
    print(f"{'Operator':<10} | {'Abs (99%)':<12} | {'Rel (99%)':<12} | {'Samples':<8}")
    print("-" * 50)
    for op_name, stats in ops_stats.items():
        if len(stats['abs']) > 0:
            p99_abs = np.percentile(stats['abs'], 99)
            p99_rel = np.percentile(stats['rel'], 99)
            count = len(stats['abs'])
            print(f"{op_name.upper():<10} | {p99_abs:.2e}     | {p99_rel:.2e}     | {count:<8}")
    print("-" * 50)

    if failed_ops:
        print(f"ERROR: numerical verification failed for: {sorted(set(failed_ops))}")
        sys.exit(1)
