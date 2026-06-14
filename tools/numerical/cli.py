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
    MaxRoiPool, RoiAlign, GridSample, LRN,
    ADD, SUB, MUL, DIV, MatMul, MatMulInteger, QLinearMatMul,
    ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd,
    ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare,
    RELU, ABS, Pow, SQRT, Conv, ConvTranspose, Col2Im, DeformConv, Attention, ConvInteger, QLinearConv, ScatterND, TensorScatter, Clip,
    Equal, Greater, Less, GreaterOrEqual, LessOrEqual,
    Gather, GatherElements, GatherND, COS, LOG, EXP, SIGMOID, TANH,
    Sin, Floor, Atan, Sign, Tan, Neg, Mod, Max, Min, Not, And, Or, Xor, IsNaN, IsInf,
    BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, BitShift,
    CumSum, CumProd, Softmax, Hardmax, LogSoftmax, NonZero, TopK, ArgMin, ArgMax, Resize, AffineGrid,
    RandomUniform, RandomUniformLike, RandomNormal, RandomNormalLike, Bernoulli, Multinomial, Dropout, Einsum,
    QuantizeLinear, DequantizeLinear, MaxUnpool, DFT, STFT, RNN, GRU, LSTM,
    Flatten, Reshape, Squeeze, Unsqueeze, Transpose, Tile, Concat, Expand, Pad, CenterCropPad, Slice, Split, Compress, ScatterElements, ConstantOfShape, EyeLike, DepthToSpace, SpaceToDepth,
    Range, OneHot, ReverseSequence, Tril, Triu, Trilu, HannWindow, HammingWindow, BlackmanWindow,
    Mean, Sum, BatchNormalization, InstanceNormalization, LayerNormalization, LpNormalization, GroupNormalization, MeanVarianceNormalization, RMSNormalization, Cast, CastLike, Ceil, Reciprocal, Softplus, Softsign, HardSigmoid,
    Det, MelWeightMatrix,
    RotaryEmbedding,
    BitCast,
    NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss, NonMaxSuppression,
    Elu, LeakyRelu, PRelu, Selu, Celu, ThresholdedRelu, Binarizer, DynamicQuantizeLinear, Unique,
    HardSwish, Swish, Shrink, Gelu, Mish,
    Round, Erf, Acos, Asin, Cosh, Sinh, Asinh, Acosh, Atanh, Identity, Where, Size,
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
        (IsInf, "isinf", [(16, 16)], ["float16"], "bool", {"detect_negative": 1, "detect_positive": 1}),
        (IsInf, "isinf", [(16, 16)], ["bfloat16"], "bool", {"detect_negative": 1, "detect_positive": 1}),
        (Identity, "identity", [(16, 16)], ["float16"], "float16"),
        (Identity, "identity", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Identity, "identity", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Identity, "identity", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Where, "where", [(16, 16), (16, 16), (16, 16)], ["bool", "float16", "float16"], "float16"),
        (Where, "where", [(16, 16), (16, 16), (16, 16)], ["bool", "bfloat16", "bfloat16"], "bfloat16"),
        (Where, "where", [(16, 16), (16, 16), (16, 16)], ["bool", "float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (Where, "where", [(16, 16), (16, 16), (16, 16)], ["bool", "float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (Size, "size", [(2, 3, 4)], ["float16"], "int64"),
        (Size, "size", [(2, 3, 4)], ["bfloat16"], "int64"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float16", "float16", "float16"], "float16"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e4m3", "float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (Mean, "mean", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e5m2", "float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float16", "float16", "float16"], "float16"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e4m3", "float8_e4m3", "float8_e4m3"], "float8_e4m3"),
        (Sum, "sum", [(8, 1, 4), (1, 3, 4), (8, 3, 1)], ["float8_e5m2", "float8_e5m2", "float8_e5m2"], "float8_e5m2"),
        (MeanVarianceNormalization, "mean_variance_normalization", [(2, 3, 2, 2)], ["float16"], "float16", {"axes": [0, 2, 3]}),
        (MeanVarianceNormalization, "mean_variance_normalization", [(2, 3, 2, 2)], ["bfloat16"], "bfloat16", {"axes": [0, 2, 3]}),
        (BatchNormalization, "batch_normalization", [(2, 3, 2, 2), (3,), (3,), (3,), (3,)], ["float16", "float16", "float16", "float16", "float16"], "float16", {"epsilon": 1e-4}),
        (BatchNormalization, "batch_normalization", [(2, 3, 2, 2), (3,), (3,), (3,), (3,)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"epsilon": 1e-4}),
        (BatchNormalization, "batch_normalization", [(2, 3, 2, 2), (3,), (3,), (3,), (3,)], ["float16", "float16", "float16", "float16", "float16"], "float16", {"epsilon": 1e-4, "momentum": 0.75, "training_mode": 1}),
        (BatchNormalization, "batch_normalization", [(2, 3, 2, 2), (3,), (3,), (3,), (3,)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"epsilon": 1e-4, "momentum": 0.75, "training_mode": 1}),
        (InstanceNormalization, "instance_normalization", [(2, 3, 2, 2), (3,), (3,)], ["float16", "float16", "float16"], "float16", {"epsilon": 1e-4}),
        (InstanceNormalization, "instance_normalization", [(2, 3, 2, 2), (3,), (3,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"epsilon": 1e-4}),
        (LayerNormalization, "layer_normalization", [(2, 3, 4), (4,), (4,)], ["float16", "float16", "float16"], "float16", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
        (LayerNormalization, "layer_normalization", [(2, 3, 4), (4,), (4,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
        (LayerNormalization, "layer_normalization", [(2, 3, 4), (3, 4), (3, 4)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"axis": 1, "epsilon": 1e-4, "stash_type": 1}),
        (LayerNormalization, "layer_normalization", [(2, 3, 4), (4,), (4,)], ["float16", "float16", "float16"], "float16", {"axis": -1, "epsilon": 1e-4, "stash_type": 1, "emit_stats": 1}),
        (LayerNormalization, "layer_normalization", [(2, 3, 4), (3, 4), (3, 4)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"axis": 1, "epsilon": 1e-4, "stash_type": 1, "emit_stats": 1}),
        (LpNormalization, "lp_normalization", [(2, 3, 2, 2)], ["float16"], "float16", {"axis": 1, "p": 2}),
        (LpNormalization, "lp_normalization", [(2, 3, 2, 2)], ["bfloat16"], "bfloat16", {"axis": 1, "p": 2}),
        (LpNormalization, "lp_normalization", [(2, 2, 3, 2)], ["bfloat16"], "bfloat16", {"axis": 2, "p": 1, "input_values": [0.0, 1.0, 0.0, -2.0, 0.0, 3.0, 4.0, 0.0, -5.0, 0.0, 6.0, 0.0, -1.0, 2.0, 3.0, -4.0, -5.0, 6.0, 0.25, -0.5, -0.75, 1.0, 1.5, -2.0]}),
        (GroupNormalization, "group_normalization", [(2, 4, 2, 2), (4,), (4,)], ["float16", "float16", "float16"], "float16", {"num_groups": 2, "epsilon": 1e-4}),
        (GroupNormalization, "group_normalization", [(2, 4, 2, 2), (4,), (4,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"num_groups": 2, "epsilon": 1e-4}),
        (RMSNormalization, "rms_normalization", [(4, 8), (8,)], ["float16", "float16"], "float16", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
        (RMSNormalization, "rms_normalization", [(4, 8), (8,)], ["bfloat16", "bfloat16"], "bfloat16", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
        (RMSNormalization, "rms_normalization", [(4, 8), (8,)], ["float8_e4m3", "float8_e4m3"], "float8_e4m3", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
        (RMSNormalization, "rms_normalization", [(4, 8), (8,)], ["float8_e5m2", "float8_e5m2"], "float8_e5m2", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
        (RotaryEmbedding, "rotary_embedding", [(2, 2, 3, 4), (6, 2), (6, 2), (2, 3)], ["float16", "float16", "float16", "int64"], "float16", {"position_ids_value": [[0, 1, 2], [3, 4, 5]], "interleaved": 0, "rotary_embedding_dim": 0}),
        (RotaryEmbedding, "rotary_embedding", [(2, 2, 3, 4), (6, 2), (6, 2), (2, 3)], ["bfloat16", "bfloat16", "bfloat16", "int64"], "bfloat16", {"position_ids_value": [[0, 1, 2], [3, 4, 5]], "interleaved": 0, "rotary_embedding_dim": 0}),
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
        (BitCast, "bitcast", [(8, 8)], ["float16"], "int16"),
        (BitCast, "bitcast", [(8, 8)], ["bfloat16"], "uint16"),
        (BitCast, "bitcast", [(8, 8)], ["float8_e4m3"], "uint8"),
        (BitCast, "bitcast", [(8, 8)], ["uint8"], "float8_e5m2"),
        (RandomUniform, "random_uniform", [], [], "float16", {"shape": [8, 8], "dtype": "float16", "low": -0.5, "high": 0.75, "seed": 211.0}),
        (RandomUniformLike, "random_uniform_like", [(8, 8)], ["bfloat16"], "bfloat16", {"low": -0.5, "high": 0.75, "seed": 223.0}),
        (RandomNormal, "random_normal", [], [], "float16", {"shape": [8, 8], "dtype": "float16", "mean": 0.5, "scale": 0.25, "seed": 227.0}),
        (RandomNormalLike, "random_normal_like", [(8, 8)], ["bfloat16"], "bfloat16", {"mean": -0.25, "scale": 0.5, "seed": 229.0}),
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
        (Binarizer, "binarizer", [(16, 16)], ["float16"], "float16", {"threshold": 0.3, "input_values": [-1.0, -0.25, 0.0, 0.3, 0.3001, 0.75, 1.0, 2.0] * 32}),
        (Binarizer, "binarizer", [(16, 16)], ["bfloat16"], "bfloat16", {"threshold": 0.3, "input_values": [-1.0, -0.25, 0.0, 0.3, 0.3001, 0.75, 1.0, 2.0] * 32}),
        (Binarizer, "binarizer", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"threshold": 0.3, "input_values": [-1.0, -0.25, 0.0, 0.3, 0.3001, 0.75, 1.0, 2.0] * 32}),
        (Binarizer, "binarizer", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"threshold": 0.3, "input_values": [-1.0, -0.25, 0.0, 0.3, 0.3001, 0.75, 1.0, 2.0] * 32}),
        (HardSwish, "hard_swish", [(16, 16)], ["float16"], "float16"),
        (HardSwish, "hard_swish", [(16, 16)], ["bfloat16"], "bfloat16"),
        (HardSwish, "hard_swish", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (HardSwish, "hard_swish", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Swish, "swish", [(16, 16)], ["float16"], "float16", {"alpha": 1.5}),
        (Swish, "swish", [(16, 16)], ["bfloat16"], "bfloat16", {"alpha": 1.5}),
        (Swish, "swish", [(16, 16)], ["float8_e4m3"], "float8_e4m3", {"alpha": 1.5}),
        (Swish, "swish", [(16, 16)], ["float8_e5m2"], "float8_e5m2", {"alpha": 1.5}),
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
        (Round, "round", [(16, 16)], ["float16"], "float16"),
        (Round, "round", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Round, "round", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Round, "round", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Erf, "erf", [(16, 16)], ["float16"], "float16"),
        (Erf, "erf", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Erf, "erf", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Erf, "erf", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Acos, "acos", [(16, 16)], ["float16"], "float16"),
        (Acos, "acos", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Acos, "acos", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Acos, "acos", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Asin, "asin", [(16, 16)], ["float16"], "float16"),
        (Asin, "asin", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Asin, "asin", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Asin, "asin", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Cosh, "cosh", [(16, 16)], ["float16"], "float16"),
        (Cosh, "cosh", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Cosh, "cosh", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Cosh, "cosh", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Sinh, "sinh", [(16, 16)], ["float16"], "float16"),
        (Sinh, "sinh", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Sinh, "sinh", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Sinh, "sinh", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Asinh, "asinh", [(16, 16)], ["float16"], "float16"),
        (Asinh, "asinh", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Asinh, "asinh", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Asinh, "asinh", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Acosh, "acosh", [(16, 16)], ["float16"], "float16"),
        (Acosh, "acosh", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Acosh, "acosh", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Acosh, "acosh", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),
        (Atanh, "atanh", [(16, 16)], ["float16"], "float16"),
        (Atanh, "atanh", [(16, 16)], ["bfloat16"], "bfloat16"),
        (Atanh, "atanh", [(16, 16)], ["float8_e4m3"], "float8_e4m3"),
        (Atanh, "atanh", [(16, 16)], ["float8_e5m2"], "float8_e5m2"),

        # ---- 混合精度矩阵、卷积、池化和 ROI ----
        (MatMul, "matmul", [(16, 32), (32, 8)], ["float16", "float16"], "float16"),
        (MatMul, "matmul", [(16, 32), (32, 8)], ["bfloat16", "bfloat16"], "bfloat16"),
        (Gemm, "gemm", [(8, 16), (16, 4), (4,)], ["float16", "float16", "float16"], "float16", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}),
        (Gemm, "gemm", [(8, 16), (16, 4), (4,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}),
        (Conv, "conv2d", [(1, 1, 5, 5), (1, 1, 3, 3), (1,)], ["float16", "float16", "float16"], "float16", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1}),
        (Conv, "conv2d", [(1, 1, 5, 5), (1, 1, 3, 3), (1,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1}),
        (ConvTranspose, "conv_transpose", [(1, 2, 4, 4), (2, 3, 3, 3), (3,)], ["float16", "float16", "float16"], "float16", {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "group": 1, "output_padding": [1, 1]}),
        (ConvTranspose, "conv_transpose", [(1, 2, 4, 4), (2, 3, 3, 3), (3,)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "group": 1, "output_padding": [1, 1]}),
        (Col2Im, "col2im", [(1, 4, 4), (2,), (2,)], ["float16", "int64", "int64"], "float16", {"image_shape_value": [3, 3], "block_shape_value": [2, 2], "pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1]}),
        (Col2Im, "col2im", [(1, 4, 4), (2,), (2,)], ["bfloat16", "int64", "int64"], "bfloat16", {"image_shape_value": [3, 3], "block_shape_value": [2, 2], "pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1]}),
        (DeformConv, "deform_conv", [(1, 2, 4, 4), (2, 2, 3, 3), (1, 18, 2, 2), (2,), (1, 9, 2, 2)], ["float16", "float16", "float16", "float16", "float16"], "float16", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1, "offset_group": 1}),
        (DeformConv, "deform_conv", [(1, 2, 4, 4), (2, 2, 3, 3), (1, 18, 2, 2), (2,), (1, 9, 2, 2)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1, "offset_group": 1}),
        (DeformConv, "deform_conv", [(1, 4, 4, 4), (4, 2, 2, 2), (1, 16, 3, 3), (4,), (1, 8, 3, 3)], ["float16", "float16", "float16", "float16", "float16"], "float16", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 2, "offset_group": 2}),
        (DeformConv, "deform_conv", [(1, 2, 6, 6), (2, 2, 2, 2), (1, 8, 4, 4), None, None], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"pads": [1, 0, 1, 0], "strides": [2, 1], "dilations": [1, 2], "group": 1, "offset_group": 1}),
        (Attention, "attention", [(1, 4, 3, 4), (1, 2, 5, 4), (1, 2, 5, 3)], ["float16", "float16", "float16"], "float16", {"is_causal": 1, "softcap": 3.0}),
        (Attention, "attention", [(1, 4, 3, 4), (1, 2, 5, 4), (1, 2, 5, 3)], ["bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"is_causal": 1, "softcap": 3.0}),
        (Attention, "attention", [(1, 4, 3, 4), (1, 2, 4, 4), (1, 2, 4, 3), (1, 1, 3, 4)], ["float16", "float16", "float16", "float16"], "float16", {"is_causal": 1, "softcap": 3.0, "attention_mask_variant": "float_bias"}),
        (Attention, "attention", [(1, 2, 2, 3), (1, 2, 4, 3), (1, 2, 4, 2), (1, 1, 1, 4)], ["bfloat16", "bfloat16", "bfloat16", "bool"], "bfloat16", {"scale": 0.5, "is_causal": 0, "attention_mask_variant": "bool_broadcast"}),
        (MaxPool, "max_pool", [(1, 2, 8, 8)], ["float16"], "float16", {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]}),
        (MaxPool, "max_pool", [(1, 2, 8, 8)], ["bfloat16"], "bfloat16", {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]}),
        (AveragePool, "average_pool", [(1, 2, 7, 7)], ["float16"], "float16", {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "count_include_pad": 1}),
        (AveragePool, "average_pool", [(1, 2, 7, 7)], ["bfloat16"], "bfloat16", {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "count_include_pad": 1}),
        (GlobalAveragePool, "global_average_pool", [(1, 3, 5, 4)], ["float16"], "float16"),
        (GlobalAveragePool, "global_average_pool", [(1, 3, 5, 4)], ["bfloat16"], "bfloat16"),
        (LRN, "lrn", [(1, 4, 2, 3)], ["float16"], "float16", {"size": 3, "alpha": 0.3, "beta": 0.5, "bias": 1.0}),
        (LRN, "lrn", [(1, 4, 2, 3)], ["bfloat16"], "bfloat16", {"size": 3, "alpha": 0.3, "beta": 0.5, "bias": 1.0}),
        (GridSample, "grid_sample", [(1, 2, 4, 5), (1, 3, 4, 2)], ["float16", "float16"], "float16", {"mode": "linear", "padding_mode": "reflection", "align_corners": 0}),
        (GridSample, "grid_sample", [(1, 2, 4, 5), (1, 3, 4, 2)], ["bfloat16", "bfloat16"], "bfloat16", {"mode": "linear", "padding_mode": "reflection", "align_corners": 0}),
        (GridSample, "grid_sample", [(1, 2, 4, 5), (1, 3, 4, 2)], ["float16", "float16"], "float16", {"mode": "nearest", "padding_mode": "border", "align_corners": 0, "grid_variant": "nearest_border"}),
        (GridSample, "grid_sample", [(1, 2, 4, 5), (1, 3, 4, 2)], ["bfloat16", "bfloat16"], "bfloat16", {"mode": "cubic", "padding_mode": "zeros", "align_corners": 1, "grid_variant": "cubic_zeros"}),
        (MaxRoiPool, "max_roi_pool", [(2, 2, 5, 5), (2, 5)], ["float16", "float16"], "float16", {"pooled_shape": [2, 3], "spatial_scale": 1.0}),
        (MaxRoiPool, "max_roi_pool", [(2, 2, 5, 5), (2, 5)], ["bfloat16", "bfloat16"], "bfloat16", {"pooled_shape": [2, 3], "spatial_scale": 1.0}),
        (MaxRoiPool, "max_roi_pool", [(2, 2, 6, 7), (3, 5)], ["bfloat16", "bfloat16"], "bfloat16", {"pooled_shape": [3, 2], "spatial_scale": 0.5, "roi_variant": "scaled_clipped"}),
        (RoiAlign, "roi_align", [(2, 1, 4, 5), (2, 4), (2,)], ["float16", "float16", "int64"], "float16", {"output_height": 2, "output_width": 3, "sampling_ratio": 2, "spatial_scale": 1.0, "mode": "avg", "coordinate_transformation_mode": "half_pixel"}),
        (RoiAlign, "roi_align", [(2, 1, 4, 5), (2, 4), (2,)], ["bfloat16", "bfloat16", "int64"], "bfloat16", {"output_height": 2, "output_width": 3, "sampling_ratio": 2, "spatial_scale": 1.0, "mode": "avg", "coordinate_transformation_mode": "half_pixel"}),
        (RoiAlign, "roi_align", [(2, 2, 5, 6), (3, 4), (3,)], ["float16", "float16", "int64"], "float16", {"output_height": 2, "output_width": 2, "sampling_ratio": 0, "spatial_scale": 0.75, "mode": "max", "coordinate_transformation_mode": "output_half_pixel", "roi_variant": "max_output_half_pixel"}),

        # ---- 混合精度归约、排序、量化、谱和循环网络 ----
        (ReduceSum, "reduce_sum", [(32, 32)], ["float16"], "float16", {"axes": None, "keepdims": 0}),
        (ReduceSum, "reduce_sum", [(32, 32)], ["bfloat16"], "bfloat16", {"axes": None, "keepdims": 0}),
        (ReduceMean, "reduce_mean", [(32, 32)], ["float16"], "float16"),
        (ReduceMean, "reduce_mean", [(32, 32)], ["bfloat16"], "bfloat16"),
        (ReduceL1, "reduce_l1", [(8, 8)], ["float16"], "float16", {"axes": None, "keepdims": 0}),
        (ReduceL1, "reduce_l1", [(8, 8)], ["bfloat16"], "bfloat16", {"axes": None, "keepdims": 0}),
        (ReduceL1, "reduce_l1", [(8, 8)], ["float8_e4m3"], "float8_e4m3", {"axes": None, "keepdims": 0}),
        (ReduceL1, "reduce_l1", [(8, 8)], ["float8_e5m2"], "float8_e5m2", {"axes": None, "keepdims": 0}),
        (ReduceL2, "reduce_l2", [(8, 8)], ["float16"], "float16", {"axes": None, "keepdims": 0}),
        (ReduceL2, "reduce_l2", [(8, 8)], ["bfloat16"], "bfloat16", {"axes": None, "keepdims": 0}),
        (ReduceL2, "reduce_l2", [(8, 8)], ["float8_e4m3"], "float8_e4m3", {"axes": None, "keepdims": 0}),
        (ReduceL2, "reduce_l2", [(8, 8)], ["float8_e5m2"], "float8_e5m2", {"axes": None, "keepdims": 0}),
        (ReduceLogSum, "reduce_log_sum", [(8, 8)], ["float16"], "float16", {"axes": None, "keepdims": 0}),
        (ReduceLogSum, "reduce_log_sum", [(8, 8)], ["bfloat16"], "bfloat16", {"axes": None, "keepdims": 0}),
        (ReduceLogSum, "reduce_log_sum", [(8, 8)], ["float8_e4m3"], "float8_e4m3", {"axes": None, "keepdims": 0}),
        (ReduceLogSum, "reduce_log_sum", [(8, 8)], ["float8_e5m2"], "float8_e5m2", {"axes": None, "keepdims": 0}),
        (ReduceLogSumExp, "reduce_log_sum_exp", [(8, 8)], ["float16"], "float16", {"axes": None, "keepdims": 0}),
        (ReduceLogSumExp, "reduce_log_sum_exp", [(8, 8)], ["bfloat16"], "bfloat16", {"axes": None, "keepdims": 0}),
        (ReduceLogSumExp, "reduce_log_sum_exp", [(8, 8)], ["float8_e4m3"], "float8_e4m3", {"axes": None, "keepdims": 0}),
        (ReduceLogSumExp, "reduce_log_sum_exp", [(8, 8)], ["float8_e5m2"], "float8_e5m2", {"axes": None, "keepdims": 0}),
        (ReduceSumSquare, "reduce_sum_square", [(8, 8)], ["float16"], "float16", {"axes": None, "keepdims": 0}),
        (ReduceSumSquare, "reduce_sum_square", [(8, 8)], ["bfloat16"], "bfloat16", {"axes": None, "keepdims": 0}),
        (ReduceSumSquare, "reduce_sum_square", [(8, 8)], ["float8_e4m3"], "float8_e4m3", {"axes": None, "keepdims": 0}),
        (ReduceSumSquare, "reduce_sum_square", [(8, 8)], ["float8_e5m2"], "float8_e5m2", {"axes": None, "keepdims": 0}),
        (Softmax, "softmax", [(4, 16)], ["float16"], "float16", {"axis": -1}),
        (Softmax, "softmax", [(4, 16)], ["bfloat16"], "bfloat16", {"axis": -1}),
        (Hardmax, "hardmax", [(4, 16)], ["float16"], "float16", {"axis": -1}),
        (Hardmax, "hardmax", [(4, 16)], ["bfloat16"], "bfloat16", {"axis": -1}),
        (Hardmax, "hardmax", [(4, 16)], ["float8_e4m3"], "float8_e4m3", {"axis": -1}),
        (Hardmax, "hardmax", [(4, 16)], ["float8_e5m2"], "float8_e5m2", {"axis": -1}),
        (LogSoftmax, "log_softmax", [(4, 16)], ["float16"], "float16", {"axis": -1}),
        (LogSoftmax, "log_softmax", [(4, 16)], ["bfloat16"], "bfloat16", {"axis": -1}),
        (LogSoftmax, "log_softmax", [(4, 16)], ["float8_e4m3"], "float8_e4m3", {"axis": -1}),
        (LogSoftmax, "log_softmax", [(4, 16)], ["float8_e5m2"], "float8_e5m2", {"axis": -1}),
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
        (CumProd, "cumprod", [(16,)], ["float16"], "float16", {"exclusive": 1, "reverse": 1}),
        (CumProd, "cumprod", [(16,)], ["bfloat16"], "bfloat16", {"exclusive": 1, "reverse": 1}),
        (CumProd, "cumprod", [(16,)], ["float8_e4m3"], "float8_e4m3", {"exclusive": 1, "reverse": 1}),
        (CumProd, "cumprod", [(16,)], ["float8_e5m2"], "float8_e5m2", {"exclusive": 1, "reverse": 1}),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["float16", "int64", "float16"], "float16"),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["bfloat16", "int64", "bfloat16"], "bfloat16"),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["float8_e4m3", "int64", "float8_e4m3"], "float8_e4m3"),
        (ScatterND, "scatternd", [(16, 16), (32, 2), (32,)], ["float8_e5m2", "int64", "float8_e5m2"], "float8_e5m2"),
        (TensorScatter, "tensor_scatter", [(2, 1, 4, 5), (2, 1, 2, 5), (2,)], ["float16", "float16", "int64"], "float16", {"axis": -2, "mode": "circular", "write_indices_value": [3, 2]}),
        (TensorScatter, "tensor_scatter", [(2, 1, 4, 5), (2, 1, 2, 5), (2,)], ["bfloat16", "bfloat16", "int64"], "bfloat16", {"axis": -2, "mode": "circular", "write_indices_value": [3, 2]}),
        (TensorScatter, "tensor_scatter", [(2, 1, 4, 5), (2, 1, 2, 5), (2,)], ["float8_e4m3", "float8_e4m3", "int64"], "float8_e4m3", {"axis": -2, "mode": "circular", "write_indices_value": [3, 2]}),
        (TensorScatter, "tensor_scatter", [(2, 1, 4, 5), (2, 1, 2, 5), (2,)], ["float8_e5m2", "float8_e5m2", "int64"], "float8_e5m2", {"axis": -2, "mode": "circular", "write_indices_value": [3, 2]}),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["float16", "float16", "float16", "int64"], "float16", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["bfloat16", "bfloat16", "bfloat16", "int64"], "bfloat16", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["float8_e4m3", "float8_e4m3", "float8_e4m3", "int64"], "float8_e4m3", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (Resize, "resize", [(1, 2, 4, 4), (0,), (0,), (4,)], ["float8_e5m2", "float8_e5m2", "float8_e5m2", "int64"], "float8_e5m2", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1, 2, 8, 8]}),
        (AffineGrid, "affine_grid", [(2, 2, 3), (4,)], ["float16", "int64"], "float16", {"size_value": [2, 1, 3, 4], "align_corners": 0}),
        (AffineGrid, "affine_grid", [(2, 2, 3), (4,)], ["bfloat16", "int64"], "bfloat16", {"size_value": [2, 1, 3, 4], "align_corners": 0}),
        (AffineGrid, "affine_grid", [(2, 2, 3), (4,)], ["float8_e4m3", "int64"], "float8_e4m3", {"size_value": [2, 1, 3, 4], "align_corners": 0}),
        (AffineGrid, "affine_grid", [(2, 2, 3), (4,)], ["float8_e5m2", "int64"], "float8_e5m2", {"size_value": [2, 1, 3, 4], "align_corners": 0}),
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
        (Squeeze, "squeeze", [(1, 2, 1, 3)], ["float16"], "float16", {"axes": [0, 2]}),
        (Squeeze, "squeeze", [(1, 2, 1, 3)], ["bfloat16"], "bfloat16", {"axes": [0, 2]}),
        (Squeeze, "squeeze", [(1, 2, 1, 3)], ["float8_e4m3"], "float8_e4m3", {"axes": [0, 2]}),
        (Squeeze, "squeeze", [(1, 2, 1, 3)], ["float8_e5m2"], "float8_e5m2", {"axes": [0, 2]}),
        (Unsqueeze, "unsqueeze", [(2, 3)], ["float16"], "float16", {"axes": [0, 2]}),
        (Unsqueeze, "unsqueeze", [(2, 3)], ["bfloat16"], "bfloat16", {"axes": [0, 2]}),
        (Unsqueeze, "unsqueeze", [(2, 3)], ["float8_e4m3"], "float8_e4m3", {"axes": [0, 2]}),
        (Unsqueeze, "unsqueeze", [(2, 3)], ["float8_e5m2"], "float8_e5m2", {"axes": [0, 2]}),
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
        (CenterCropPad, "center_crop_pad", [(2, 3, 4), (2,)], ["float16", "int64"], "float16", {"axes": [-1, 1], "target_shape": [5, 2]}),
        (CenterCropPad, "center_crop_pad", [(2, 3, 4), (2,)], ["bfloat16", "int64"], "bfloat16", {"axes": [-1, 1], "target_shape": [5, 2]}),
        (CenterCropPad, "center_crop_pad", [(2, 3, 4), (2,)], ["float8_e4m3", "int64"], "float8_e4m3", {"axes": [-1, 1], "target_shape": [5, 2]}),
        (CenterCropPad, "center_crop_pad", [(2, 3, 4), (2,)], ["float8_e5m2", "int64"], "float8_e5m2", {"axes": [-1, 1], "target_shape": [5, 2]}),
        (Slice, "slice", [(2, 4, 3), (3,), (3,), (3,), (3,)], ["float16", "int64", "int64", "int64", "int64"], "float16", {"starts_value": [0, 1, 0], "ends_value": [2, 4, 3], "axes_value": [0, 1, 2], "steps_value": [1, 2, 1]}),
        (Slice, "slice", [(2, 4, 3), (3,), (3,), (3,), (3,)], ["bfloat16", "int64", "int64", "int64", "int64"], "bfloat16", {"starts_value": [0, 1, 0], "ends_value": [2, 4, 3], "axes_value": [0, 1, 2], "steps_value": [1, 2, 1]}),
        (Slice, "slice", [(2, 4, 3), (3,), (3,), (3,), (3,)], ["float8_e4m3", "int64", "int64", "int64", "int64"], "float8_e4m3", {"starts_value": [0, 1, 0], "ends_value": [2, 4, 3], "axes_value": [0, 1, 2], "steps_value": [1, 2, 1]}),
        (Slice, "slice", [(2, 4, 3), (3,), (3,), (3,), (3,)], ["float8_e5m2", "int64", "int64", "int64", "int64"], "float8_e5m2", {"starts_value": [0, 1, 0], "ends_value": [2, 4, 3], "axes_value": [0, 1, 2], "steps_value": [1, 2, 1]}),
        (Compress, "compress", [(2, 4, 3), (4,)], ["float16", "bool"], "float16", {"axis": 1, "condition_value": [True, False, True, False]}),
        (Compress, "compress", [(2, 4, 3), (4,)], ["bfloat16", "bool"], "bfloat16", {"axis": 1, "condition_value": [True, False, True, False]}),
        (Compress, "compress", [(2, 4, 3), (4,)], ["float8_e4m3", "bool"], "float8_e4m3", {"axis": 1, "condition_value": [True, False, True, False]}),
        (Compress, "compress", [(2, 4, 3), (4,)], ["float8_e5m2", "bool"], "float8_e5m2", {"axis": 1, "condition_value": [True, False, True, False]}),
        (ScatterElements, "scatter_elements", [(3, 4), (3, 4), (3, 4)], ["float16", "int64", "float16"], "float16", {"axis": 1, "reduction": "none"}),
        (ScatterElements, "scatter_elements", [(3, 4), (3, 4), (3, 4)], ["bfloat16", "int64", "bfloat16"], "bfloat16", {"axis": 1, "reduction": "none"}),
        (ScatterElements, "scatter_elements", [(3, 4), (3, 4), (3, 4)], ["float8_e4m3", "int64", "float8_e4m3"], "float8_e4m3", {"axis": 1, "reduction": "none"}),
        (ScatterElements, "scatter_elements", [(3, 4), (3, 4), (3, 4)], ["float8_e5m2", "int64", "float8_e5m2"], "float8_e5m2", {"axis": 1, "reduction": "none"}),
        (DepthToSpace, "depth_to_space", [(1, 8, 2, 3)], ["float16"], "float16", {"blocksize": 2, "mode": "DCR"}),
        (DepthToSpace, "depth_to_space", [(1, 8, 2, 3)], ["bfloat16"], "bfloat16", {"blocksize": 2, "mode": "DCR"}),
        (DepthToSpace, "depth_to_space", [(1, 8, 2, 3)], ["float8_e4m3"], "float8_e4m3", {"blocksize": 2, "mode": "DCR"}),
        (DepthToSpace, "depth_to_space", [(1, 8, 2, 3)], ["float8_e5m2"], "float8_e5m2", {"blocksize": 2, "mode": "DCR"}),
        (SpaceToDepth, "space_to_depth", [(1, 2, 4, 6)], ["float16"], "float16", {"blocksize": 2}),
        (SpaceToDepth, "space_to_depth", [(1, 2, 4, 6)], ["bfloat16"], "bfloat16", {"blocksize": 2}),
        (SpaceToDepth, "space_to_depth", [(1, 2, 4, 6)], ["float8_e4m3"], "float8_e4m3", {"blocksize": 2}),
        (SpaceToDepth, "space_to_depth", [(1, 2, 4, 6)], ["float8_e5m2"], "float8_e5m2", {"blocksize": 2}),
        (Split, "split", [(2, 6), (3,)], ["float16", "int64"], "float16", {"axis": 1, "split_value": [1, 3, 2], "num_outputs": 3, "input_values": [-3.0, -2.5, -1.0, -0.25, 0.0, 0.5, 1.25, 2.0, 3.5, 4.0, 5.25, 6.0]}),
        (Unique, "unique", [(8,)], ["float16"], "float16", {"sorted": 1, "input_values": [2.0, -1.0, 2.0, 0.5, -1.0, 3.0, 0.5, 4.0]}),
        (QuantizeLinear, "quantize_linear", [(32, 32), (1,), (1,)], ["float16", "float16", "int8"], "int8"),
        (QuantizeLinear, "quantize_linear", [(32, 32), (1,), (1,)], ["bfloat16", "bfloat16", "int8"], "int8"),
        (QuantizeLinear, "quantize_linear", [(1, 3, 4), (3,), (3,)], ["float16", "float16", "uint8"], "uint8", {"axis": 1, "input_values": [-20.0, -0.05, 0.0, 0.05, -1.0, 0.0, 1.25, 25.0, -10.0, -0.5, 2.5, 100.0], "scale_values": [0.1, 0.25, 0.5], "zero_point_values": [0, 128, 250]}),
        (QuantizeLinear, "quantize_linear", [(2, 3, 4), (4,), (1,)], ["float16", "float16", "int8"], "int8", {"axis": -1, "omit_zero_point": 1, "input_values": [-25.0, -1.0, -0.49, 0.49, -12.8, -0.75, 0.0, 0.75, -2.5, -0.125, 0.125, 2.5, -100.0, -3.0, 3.0, 100.0, -6.4, -1.25, 1.25, 6.4, -0.05, 0.05, 10.0, -10.0], "scale_values": [0.1, 0.25, 0.5, 1.25]}),
        (QuantizeLinear, "quantize_linear", [(2, 3, 4), (2, 2, 4), (2, 2, 4)], ["float16", "float16", "int8"], "int8", {"axis": 1, "block_size": 2, "input_values": [-2.333, -2.0, -1.667, -1.333, -1.0, -0.667, -0.333, 0.0, 0.333, 0.667, 1.0, 1.333, 1.667, 2.0, 2.333, 2.667, 3.0, 3.333, 3.667, 4.0, 4.333, 4.667, 5.0, 5.333], "scale_values": [0.1, 0.2, 0.25, 0.5, 0.3, 0.4, 0.6, 0.8, 0.15, 0.35, 0.45, 0.55, 0.25, 0.5, 0.75, 1.0], "zero_point_values": [-5, -4, -3, -2, 1, 2, 3, 4, -8, -6, -4, -2, 2, 4, 6, 8]}),
        (QuantizeLinear, "quantize_linear", [(2, 3, 5), (2, 3, 3), (2, 3, 3)], ["float16", "float16", "int8"], "int8", {"axis": -1, "block_size": 2, "input_values": [-3.0, -2.5, -1.0, -0.25, 0.0, 0.5, 1.25, 2.0, 3.5, 4.0, -4.0, -1.5, -0.5, 0.25, 1.75, 2.5, -2.25, -0.75, 0.75, 2.25, 5.0, -5.0, 6.0, -6.0, 0.125, -0.125, 7.0, -7.0, 8.0, -8.0], "scale_values": [0.1, 0.2, 0.4, 0.15, 0.3, 0.6, 0.25, 0.5, 0.75, 0.12, 0.24, 0.48, 0.18, 0.36, 0.72, 0.2, 0.45, 0.9], "zero_point_values": [-5, -3, -1, 0, 2, 4, -8, -4, 0, 1, 3, 5, -7, -5, -3, 2, 6, 10]}),
        (QuantizeLinear, "quantize_linear", [(8,), (1,), (1,)], ["float16", "float16", "int16"], "int16", {"axis": 0, "input_values": [-40000.0, -123.4, -0.5, 0.0, 0.5, 123.4, 32767.4, 40000.0], "scale_values": [1.0], "zero_point_values": [0]}),
        (DequantizeLinear, "dequantize_linear", [(32, 32), (1,), (1,)], ["int8", "float16", "int8"], "float16"),
        (DequantizeLinear, "dequantize_linear", [(32, 32), (1,), (1,)], ["int8", "bfloat16", "int8"], "bfloat16"),
        (DequantizeLinear, "dequantize_linear", [(1, 3, 4), (3,), (3,)], ["uint8", "bfloat16", "uint8"], "bfloat16", {"axis": 1, "input_values": [0, 1, 128, 255, 0, 128, 129, 255, 0, 250, 251, 255], "scale_values": [0.1, 0.25, 0.5], "zero_point_values": [0, 128, 250]}),
        (DequantizeLinear, "dequantize_linear", [(2, 3, 4), (4,), (1,)], ["int8", "bfloat16", "int8"], "bfloat16", {"axis": -1, "omit_zero_point": 1, "input_values": [-128, -4, 0, 127, -64, -3, 3, 64, -25, -1, 1, 25, -10, -2, 2, 10, -5, -1, 1, 5, -100, -8, 8, 100], "scale_values": [0.1, 0.25, 0.5, 1.25]}),
        (DequantizeLinear, "dequantize_linear", [(2, 3, 4), (2, 2, 4), (2, 2, 4)], ["int8", "bfloat16", "int8"], "bfloat16", {"axis": 1, "block_size": 2, "input_values": [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, -11, -7, -3, 1, 3, 5, 7, 9, -9, -5, -1, 11], "scale_values": [0.1, 0.2, 0.25, 0.5, 0.3, 0.4, 0.6, 0.8, 0.15, 0.35, 0.45, 0.55, 0.25, 0.5, 0.75, 1.0], "zero_point_values": [-5, -4, -3, -2, 1, 2, 3, 4, -8, -6, -4, -2, 2, 4, 6, 8]}),
        (DequantizeLinear, "dequantize_linear", [(2, 3, 5), (2, 3, 3), (2, 3, 3)], ["int8", "bfloat16", "int8"], "bfloat16", {"axis": -1, "block_size": 2, "input_values": [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, -11, -7, -3, 1, 3, 5, 7, 9, -9, -5, -1, 11, 13, -13, 15, -15, 31, -31], "scale_values": [0.1, 0.2, 0.4, 0.15, 0.3, 0.6, 0.25, 0.5, 0.75, 0.12, 0.24, 0.48, 0.18, 0.36, 0.72, 0.2, 0.45, 0.9], "zero_point_values": [-5, -3, -1, 0, 2, 4, -8, -4, 0, 1, 3, 5, -7, -5, -3, 2, 6, 10]}),
        (DequantizeLinear, "dequantize_linear", [(8,), (1,), (1,)], ["uint16", "bfloat16", "uint16"], "bfloat16", {"axis": 0, "input_values": [0, 5, 10, 255, 1024, 4096, 40000, 65535], "scale_values": [0.25], "zero_point_values": [5]}),
        (QLinearMatMul, "qlinear_matmul", [(4, 6), (4,), (4,), (6, 5), (5,), (5,), (1,), (1,)], ["uint8", "bfloat16", "uint8", "uint8", "bfloat16", "uint8", "bfloat16", "uint8"], "uint8"),
        (QLinearConv, "qlinear_conv", [(1, 2, 5, 5), (1,), (1,), (2, 2, 3, 3), (2,), (2,), (1,), (1,)], ["uint8", "bfloat16", "uint8", "uint8", "bfloat16", "uint8", "bfloat16", "uint8"], "uint8", {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "group": 1}),
        (Range, "range", [(1,), (1,), (1,)], ["float16", "float16", "float16"], "float16", {"start_value": -2.0, "limit_value": 3.0, "delta_value": 0.75}),
        (OneHot, "one_hot", [(2, 3), (), (2,)], ["int64", "int64", "float16"], "float16", {"axis": -1, "depth_value": 4, "values_value": [-0.5, 2.0]}),
        (ReverseSequence, "reverse_sequence", [(4, 3, 2), (3,)], ["float16", "int64"], "float16", {"time_axis": 0, "batch_axis": 1, "sequence_lens_value": [4, 2, 3]}),
        (Det, "det", [(2, 3, 3)], ["float16"], "float16"),
        (MelWeightMatrix, "mel_weight_matrix", [(1,), (1,), (1,), (1,), (1,)], ["int64", "int64", "int64", "float32", "float32"], "bfloat16", {"num_mel_bins_value": 4, "dft_length_value": 10, "sample_rate_value": 16000, "lower_edge_hertz_value": 20.0, "upper_edge_hertz_value": 7600.0}),
        (Tril, "tril", [(3, 4), ()], ["float16", "int64"], "float16", {"k_value": -1}),
        (Triu, "triu", [(3, 4), ()], ["bfloat16", "int64"], "bfloat16", {"k_value": 1}),
        (Trilu, "trilu", [(3, 4), ()], ["float16", "int64"], "float16", {"upper": 0, "k_value": 0}),
        (HannWindow, "hann_window", [()], ["int64"], "float16", {"periodic": 0, "window_size_value": 8}),
        (HammingWindow, "hamming_window", [()], ["int64"], "bfloat16", {"periodic": 1, "window_size_value": 9}),
        (BlackmanWindow, "blackman_window", [()], ["int64"], "float16", {"periodic": 0, "window_size_value": 10}),
        (DFT, "dft", [(1, 4, 1), ()], ["float16", "int64"], "float16", {"axis": 1, "onesided": 1, "inverse": 0, "dft_length_value": 4}),
        (DFT, "dft", [(1, 4, 1), ()], ["bfloat16", "int64"], "bfloat16", {"axis": 1, "onesided": 1, "inverse": 0, "dft_length_value": 4}),
        (DFT, "dft", [(1, 4, 2), ()], ["float16", "int64"], "float16", {"axis": 1, "onesided": 0, "inverse": 0, "dft_length_value": 4, "dft_variant": "complex_full"}),
        (DFT, "dft", [(1, 3, 2), ()], ["bfloat16", "int64"], "bfloat16", {"axis": 1, "onesided": 1, "inverse": 1, "dft_length_value": 4, "dft_variant": "inverse_onesided"}),
        (DFT, "dft", [(2, 3, 4, 1), ()], ["float16", "int64"], "float16", {"axis": 1, "onesided": 0, "inverse": 0, "dft_length_value": 5, "dft_variant": "high_rank_axis"}),
        (STFT, "stft", [(1, 4, 1), (), (2,), ()], ["float16", "int64", "float16", "int64"], "float16", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 2}),
        (STFT, "stft", [(1, 4, 1), (), (2,), ()], ["bfloat16", "int64", "bfloat16", "int64"], "bfloat16", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 2}),
        (STFT, "stft", [(1, 5, 1), (), (3,), ()], ["float16", "int64", "float16", "int64"], "float16", {"onesided": 0, "frame_step_value": 2, "frame_length_value": 3, "stft_variant": "real_window_full"}),
        (STFT, "stft", [(1, 5, 2), (), (3,), ()], ["bfloat16", "int64", "bfloat16", "int64"], "bfloat16", {"onesided": 0, "frame_step_value": 2, "frame_length_value": 3, "stft_variant": "complex_no_window_full"}),
        (STFT, "stft", [(2, 2, 6, 1), (), (4,), ()], ["bfloat16", "int64", "bfloat16", "int64"], "bfloat16", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 4, "stft_variant": "high_rank_prefix"}),
        (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0}),
        (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "reverse", "layout": 0}),
        (RNN, "rnn", [(2, 3, 2), (2, 2, 2), (2, 2, 2), (2, 4), (2,), (2, 2, 2)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "int64", "bfloat16"], "bfloat16", {"hidden_size": 2, "direction": "bidirectional", "layout": 1}),
        (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "activations": ["Relu"], "clip": 0.35}),
        (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "sequence_lens_value": [0, 2]}),
        (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 1}),
        (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float16", "float16", "float16", "float16", "int64", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 0}),
        (GRU, "gru", [(2, 3, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "int64", "bfloat16"], "bfloat16", {"hidden_size": 2, "direction": "reverse", "layout": 1, "linear_before_reset": 1}),
        (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "int64", "bfloat16"], "bfloat16", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 0, "activations": ["HardSigmoid", "ScaledTanh"], "activation_alpha": [0.25, 1.1], "activation_beta": [0.45, 0.7], "clip": 0.4}),
        (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "int64", "bfloat16"], "bfloat16", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 1, "sequence_lens_value": [0, 2]}),
        (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float16", "float16", "float16", "float16", "int64", "float16", "float16", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 1}),
        (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float16", "float16", "float16", "float16", "int64", "float16", "float16", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 0}),
        (LSTM, "lstm", [(2, 3, 2), (2, 8, 2), (2, 8, 2), (2, 16), (2,), (2, 2, 2), (2, 2, 2), (2, 6)], ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "int64", "bfloat16", "bfloat16", "bfloat16"], "bfloat16", {"hidden_size": 2, "direction": "bidirectional", "layout": 1, "input_forget": 0}),
        (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float16", "float16", "float16", "float16", "int64", "float16", "float16", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 0, "activations": ["HardSigmoid", "Tanh", "Relu"], "activation_alpha": [0.25], "activation_beta": [0.45], "clip": 0.35}),
        (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float16", "float16", "float16", "float16", "int64", "float16", "float16", "float16"], "float16", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 1, "sequence_lens_value": [0, 2]}),
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
    (Col2Im, "col2im", [(1, 4, 4), (2,), (2,)], ["float32", "int64", "int64"], "float32", {"image_shape_value": [3, 3], "block_shape_value": [2, 2], "pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1]}),
    (DeformConv, "deform_conv", [(1, 2, 4, 4), (2, 2, 3, 3), (1, 18, 2, 2), (2,), (1, 9, 2, 2)], ["float32", "float32", "float32", "float32", "float32"], "float32", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1, "offset_group": 1}),
    (DeformConv, "deform_conv", [(1, 4, 4, 4), (4, 2, 2, 2), (1, 16, 3, 3), (4,), (1, 8, 3, 3)], ["float32", "float32", "float32", "float32", "float32"], "float32", {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 2, "offset_group": 2}),
    (DeformConv, "deform_conv", [(1, 2, 6, 6), (2, 2, 2, 2), (1, 8, 4, 4), None, None], ["float32", "float32", "float32", "float32", "float32"], "float32", {"pads": [1, 0, 1, 0], "strides": [2, 1], "dilations": [1, 2], "group": 1, "offset_group": 1}),
    (Attention, "attention", [(1, 4, 3, 4), (1, 2, 5, 4), (1, 2, 5, 3)], ["float32", "float32", "float32"], "float32", {"is_causal": 1, "softcap": 3.0}),
    (Attention, "attention", [(1, 4, 3, 4), (1, 2, 4, 4), (1, 2, 4, 3), (1, 1, 3, 4)], ["float32", "float32", "float32", "float32"], "float32", {"is_causal": 1, "softcap": 3.0, "attention_mask_variant": "float_bias"}),
    (Attention, "attention", [(1, 2, 2, 3), (1, 2, 4, 3), (1, 2, 4, 2), (1, 1, 1, 4)], ["float32", "float32", "float32", "bool"], "float32", {"scale": 0.5, "is_causal": 0, "attention_mask_variant": "bool_broadcast"}),

    # ---- Softmax ----
    (Softmax, "softmax",[(4, 64)], ["float32"], "float32", {"axis":-1}),
    (Hardmax, "hardmax",[(4, 64)], ["float32"], "float32", {"axis":-1}),
    (LogSoftmax, "log_softmax",[(4, 64)], ["float32"], "float32", {"axis":-1}),

    # ---- Gemm ----
    (Gemm, "gemm",[(16, 32), (32, 8), (8,)], ["float32", "float32", "float32"], "float32",{"alpha":1.0, "beta":1.0, "transA":0, "transB":0}),

    # ---- MaxPool ----
    (MaxPool, "max_pool",[(1, 2, 16, 16)], ["float32"], "float32",{"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),
    (AveragePool, "average_pool",[(1, 2, 7, 7)], ["float32"], "float32",{"kernel_shape":[3,3], "pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "count_include_pad":1}),
    (LpPool, "lp_pool",[(1, 2, 7, 7)], ["float32"], "float32",{"kernel_shape":[3,3], "pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "p":2}),
    (GlobalAveragePool, "global_average_pool",[(1, 3, 5, 4)], ["float32"], "float32"),
    (GlobalMaxPool, "global_max_pool",[(1, 3, 5, 4)], ["float32"], "float32"),
    (GlobalLpPool, "global_lp_pool",[(1, 3, 5, 4)], ["float32"], "float32", {"p": 2}),
    (LRN, "lrn", [(1, 4, 2, 3)], ["float32"], "float32", {"size": 3, "alpha": 0.3, "beta": 0.5, "bias": 1.0}),
    (MaxUnpool, "max_unpool",[(1, 1, 2, 2), (1, 1, 2, 2)], ["float32", "int64"], "float32",{"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),
    (MaxRoiPool, "max_roi_pool",[(2, 2, 5, 5), (2, 5)], ["float32", "float32"], "float32", {"pooled_shape":[2, 3], "spatial_scale":1.0}),
    (MaxRoiPool, "max_roi_pool",[(2, 2, 6, 7), (3, 5)], ["float32", "float32"], "float32", {"pooled_shape":[3, 2], "spatial_scale":0.5, "roi_variant":"scaled_clipped"}),
    (GridSample, "grid_sample", [(1, 2, 4, 5), (1, 3, 4, 2)], ["float32", "float32"], "float32", {"mode": "linear", "padding_mode": "reflection", "align_corners": 0}),
    (GridSample, "grid_sample", [(1, 2, 4, 5), (1, 3, 4, 2)], ["float32", "float32"], "float32", {"mode": "nearest", "padding_mode": "border", "align_corners": 0, "grid_variant": "nearest_border"}),
    (GridSample, "grid_sample", [(1, 2, 4, 5), (1, 3, 4, 2)], ["float32", "float32"], "float32", {"mode": "cubic", "padding_mode": "zeros", "align_corners": 1, "grid_variant": "cubic_zeros"}),
    (RoiAlign, "roi_align",[(2, 1, 4, 5), (2, 4), (2,)], ["float32", "float32", "int64"], "float32", {"output_height":2, "output_width":3, "sampling_ratio":2, "spatial_scale":1.0, "mode":"avg", "coordinate_transformation_mode":"half_pixel"}),
    (RoiAlign, "roi_align",[(2, 2, 5, 6), (3, 4), (3,)], ["float32", "float32", "int64"], "float32", {"output_height":2, "output_width":2, "sampling_ratio":0, "spatial_scale":0.75, "mode":"max", "coordinate_transformation_mode":"output_half_pixel", "roi_variant":"max_output_half_pixel"}),

    (Equal,   "equal",   [(64,64), (64,64)], ["float32", "float32"], "bool"),
    (Greater, "greater", [(64,64), (64,64)], ["float32", "float32"], "bool"),
    (Less,    "less",    [(64,64), (64,64)], ["float32", "float32"], "bool"),

    (Clip, "clip",[(64,64), (1,), (1,)],["float32", "float32", "float32"],"float32"),

    (QuantizeLinear, "quantize_linear", [(64,64), (1,), (1,)], ["float32", "float32", "int8"], "int8"),
    (QuantizeLinear, "quantize_linear", [(1, 3, 4), (3,), (3,)], ["float32", "float32", "uint8"], "uint8", {"axis": 1, "input_values": [-20.0, -0.05, 0.0, 0.05, -1.0, 0.0, 1.25, 25.0, -10.0, -0.5, 2.5, 100.0], "scale_values": [0.1, 0.25, 0.5], "zero_point_values": [0, 128, 250]}),
    (QuantizeLinear, "quantize_linear", [(2, 3, 4), (4,), (1,)], ["float32", "float32", "int8"], "int8", {"axis": -1, "omit_zero_point": 1, "input_values": [-25.0, -1.0, -0.49, 0.49, -12.8, -0.75, 0.0, 0.75, -2.5, -0.125, 0.125, 2.5, -100.0, -3.0, 3.0, 100.0, -6.4, -1.25, 1.25, 6.4, -0.05, 0.05, 10.0, -10.0], "scale_values": [0.1, 0.25, 0.5, 1.25]}),
    (QuantizeLinear, "quantize_linear", [(2, 3, 4), (2, 2, 4), (2, 2, 4)], ["float32", "float32", "int8"], "int8", {"axis": 1, "block_size": 2, "input_values": [-2.333, -2.0, -1.667, -1.333, -1.0, -0.667, -0.333, 0.0, 0.333, 0.667, 1.0, 1.333, 1.667, 2.0, 2.333, 2.667, 3.0, 3.333, 3.667, 4.0, 4.333, 4.667, 5.0, 5.333], "scale_values": [0.1, 0.2, 0.25, 0.5, 0.3, 0.4, 0.6, 0.8, 0.15, 0.35, 0.45, 0.55, 0.25, 0.5, 0.75, 1.0], "zero_point_values": [-5, -4, -3, -2, 1, 2, 3, 4, -8, -6, -4, -2, 2, 4, 6, 8]}),
    (QuantizeLinear, "quantize_linear", [(2, 3, 5), (2, 3, 3), (2, 3, 3)], ["float32", "float32", "int8"], "int8", {"axis": -1, "block_size": 2, "input_values": [-3.0, -2.5, -1.0, -0.25, 0.0, 0.5, 1.25, 2.0, 3.5, 4.0, -4.0, -1.5, -0.5, 0.25, 1.75, 2.5, -2.25, -0.75, 0.75, 2.25, 5.0, -5.0, 6.0, -6.0, 0.125, -0.125, 7.0, -7.0, 8.0, -8.0], "scale_values": [0.1, 0.2, 0.4, 0.15, 0.3, 0.6, 0.25, 0.5, 0.75, 0.12, 0.24, 0.48, 0.18, 0.36, 0.72, 0.2, 0.45, 0.9], "zero_point_values": [-5, -3, -1, 0, 2, 4, -8, -4, 0, 1, 3, 5, -7, -5, -3, 2, 6, 10]}),
    (QuantizeLinear, "quantize_linear", [(8,), (1,), (1,)], ["float32", "float32", "uint16"], "uint16", {"axis": 0, "input_values": [-10.0, -0.5, 0.0, 0.5, 123.4, 40000.0, 70000.0, 100000.0], "scale_values": [1.0], "zero_point_values": [5]}),
    (QuantizeLinear, "quantize_linear", [(1,), (1,), (1,)], ["float32", "float32", "int8"], "int8", {"axis": 0, "precision": 11, "input_values": [-12.75], "scale_values": [0.1], "zero_point_values": [0]}),
    (DequantizeLinear, "dequantize_linear", [(64,64), (1,), (1,)], ["int8", "float32", "int8"], "float32"),
    (DequantizeLinear, "dequantize_linear", [(1, 3, 4), (3,), (3,)], ["uint8", "float32", "uint8"], "float32", {"axis": 1, "input_values": [0, 1, 128, 255, 0, 128, 129, 255, 0, 250, 251, 255], "scale_values": [0.1, 0.25, 0.5], "zero_point_values": [0, 128, 250]}),
    (DequantizeLinear, "dequantize_linear", [(2, 3, 4), (4,), (1,)], ["int8", "float32", "int8"], "float32", {"axis": -1, "omit_zero_point": 1, "input_values": [-128, -4, 0, 127, -64, -3, 3, 64, -25, -1, 1, 25, -10, -2, 2, 10, -5, -1, 1, 5, -100, -8, 8, 100], "scale_values": [0.1, 0.25, 0.5, 1.25]}),
    (DequantizeLinear, "dequantize_linear", [(2, 3, 4), (2, 2, 4), (2, 2, 4)], ["int8", "float32", "int8"], "float32", {"axis": 1, "block_size": 2, "input_values": [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, -11, -7, -3, 1, 3, 5, 7, 9, -9, -5, -1, 11], "scale_values": [0.1, 0.2, 0.25, 0.5, 0.3, 0.4, 0.6, 0.8, 0.15, 0.35, 0.45, 0.55, 0.25, 0.5, 0.75, 1.0], "zero_point_values": [-5, -4, -3, -2, 1, 2, 3, 4, -8, -6, -4, -2, 2, 4, 6, 8]}),
    (DequantizeLinear, "dequantize_linear", [(2, 3, 5), (2, 3, 3), (2, 3, 3)], ["int8", "float32", "int8"], "float32", {"axis": -1, "block_size": 2, "input_values": [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, -11, -7, -3, 1, 3, 5, 7, 9, -9, -5, -1, 11, 13, -13, 15, -15, 31, -31], "scale_values": [0.1, 0.2, 0.4, 0.15, 0.3, 0.6, 0.25, 0.5, 0.75, 0.12, 0.24, 0.48, 0.18, 0.36, 0.72, 0.2, 0.45, 0.9], "zero_point_values": [-5, -3, -1, 0, 2, 4, -8, -4, 0, 1, 3, 5, -7, -5, -3, 2, 6, 10]}),
    (DequantizeLinear, "dequantize_linear", [(8,), (1,), (1,)], ["int16", "float32", "int16"], "float32", {"axis": 0, "input_values": [-32768, -123, -1, 0, 1, 123, 32760, 32767], "scale_values": [0.5], "zero_point_values": [-3]}),
    (DequantizeLinear, "dequantize_linear", [(8,), (1,), (1,)], ["int32", "float32", "int32"], "float32", {"axis": 0, "input_values": [-2147483648, -65536, -1024, 0, 1024, 65536, 123456789, 2147483647], "scale_values": [0.25], "zero_point_values": [-17]}),

    (SQRT, "sqrt", [(64, 64)], ["float32"], "float32"),

    (Pow, "pow", [(64,64), (64,64)], ["float32", "float32"], "float32"),

    (MatMul, "matmul",[(32, 64), (64,16)],["float32", "float32"],"float32"),
    (MatMulInteger, "matmul_integer", [(4, 6), (6, 5), (4,), (5,)], ["uint8", "int8", "uint8", "int8"], "int32"),
    (QLinearMatMul, "qlinear_matmul", [(4, 6), (4,), (4,), (6, 5), (5,), (5,), (1,), (1,)], ["uint8", "float32", "uint8", "uint8", "float32", "uint8", "float32", "uint8"], "uint8"),

    (ReduceMean, "reduce_mean",[(32, 64)],["float32"], "float32"),

    (Gather, "gather",[(32, 64), (8,)],["float32", "int64"],"float32",{"axis": 0}),

    (ScatterND, "scatternd",[(32, 64), (16, 2), (16,)],["float32", "int64", "float32"],"float32"), 
    (TensorScatter, "tensor_scatter", [(2, 1, 4, 5), (2, 1, 2, 5), (2,)], ["float32", "float32", "int64"], "float32", {"axis": -2, "mode": "linear", "write_indices_value": [1, 2]}),

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
    (ReduceL1, "reduce_l1", [(8,8)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceL2, "reduce_l2", [(8,8)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceLogSum, "reduce_log_sum", [(8,8)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceLogSumExp, "reduce_log_sum_exp", [(8,8)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceSumSquare, "reduce_sum_square", [(8,8)], ["float32"], "float32", {"axes":None, "keepdims":0}),

    # 逻辑（bool 输入/输出）
    (Not, "not", [(256,256)], ["bool"], "bool"),
    (And, "and", [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (Or,  "or",  [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (Xor, "xor", [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (BitwiseAnd, "bitwise_and", [(4, 4), (4, 4)], ["int32", "int32"], "int32"),
    (BitwiseOr, "bitwise_or", [(4, 4), (4, 4)], ["int32", "int32"], "int32"),
    (BitwiseXor, "bitwise_xor", [(4, 4), (4, 4)], ["int32", "int32"], "int32"),
    (BitwiseNot, "bitwise_not", [(4, 4)], ["int32"], "int32"),
    (BitShift, "bit_shift", [(4, 4), (4, 4)], ["int32", "int32"], "int32", {"direction": "LEFT"}),
    (BitShift, "bit_shift", [(4, 4), (4, 4)], ["int32", "int32"], "int32", {"direction": "RIGHT"}),
    (GreaterOrEqual, "greater_or_equal", [(256,256), (256,256)], ["float32", "float32"], "bool"),
    (LessOrEqual, "less_or_equal", [(256,256), (256,256)], ["float32", "float32"], "bool"),
    (IsInf, "isinf", [(64, 64)], ["float32"], "bool", {"detect_negative": 1, "detect_positive": 1}),
    (Identity, "identity", [(64, 64)], ["float32"], "float32"),
    (Where, "where", [(64, 64), (64, 64), (64, 64)], ["bool", "float32", "float32"], "float32"),
    (Size, "size", [(2, 3, 4)], ["float32"], "int64"),
    (Mean, "mean", [(16, 1, 8), (1, 4, 8), (16, 4, 1)], ["float32", "float32", "float32"], "float32"),
    (Sum, "sum", [(16, 1, 8), (1, 4, 8), (16, 4, 1)], ["float32", "float32", "float32"], "float32"),
    (MeanVarianceNormalization, "mean_variance_normalization", [(2, 3, 2, 2)], ["float32"], "float32", {"axes": [0, 2, 3]}),
    (BatchNormalization, "batch_normalization", [(2, 3, 2, 2), (3,), (3,), (3,), (3,)], ["float32", "float32", "float32", "float32", "float32"], "float32", {"epsilon": 1e-4}),
    (BatchNormalization, "batch_normalization", [(2, 3, 2, 2), (3,), (3,), (3,), (3,)], ["float32", "float32", "float32", "float32", "float32"], "float32", {"epsilon": 1e-4, "momentum": 0.75, "training_mode": 1}),
    (InstanceNormalization, "instance_normalization", [(2, 3, 2, 2), (3,), (3,)], ["float32", "float32", "float32"], "float32", {"epsilon": 1e-4}),
    (LayerNormalization, "layer_normalization", [(2, 3, 4), (4,), (4,)], ["float32", "float32", "float32"], "float32", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
    (LayerNormalization, "layer_normalization", [(2, 3, 4), (3, 4), (3, 4)], ["float32", "float32", "float32"], "float32", {"axis": 1, "epsilon": 1e-4, "stash_type": 1}),
    (LayerNormalization, "layer_normalization", [(2, 3, 4), (3, 4), (3, 4)], ["float32", "float32", "float32"], "float32", {"axis": 1, "epsilon": 1e-4, "stash_type": 1, "emit_stats": 1}),
    (LpNormalization, "lp_normalization", [(2, 3, 2, 2)], ["float32"], "float32", {"axis": 1, "p": 2}),
    (LpNormalization, "lp_normalization", [(2, 3, 2, 2)], ["float32"], "float32", {"axis": 1, "p": 1}),
    (LpNormalization, "lp_normalization", [(2, 3, 2)], ["float32"], "float32", {"axis": -1, "p": 2, "input_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
    (GroupNormalization, "group_normalization", [(2, 4, 2, 2), (4,), (4,)], ["float32", "float32", "float32"], "float32", {"num_groups": 2, "epsilon": 1e-4}),
    (RMSNormalization, "rms_normalization", [(4, 8), (8,)], ["float32", "float32"], "float32", {"axis": -1, "epsilon": 1e-4, "stash_type": 1}),
    (RotaryEmbedding, "rotary_embedding", [(2, 2, 3, 4), (6, 2), (6, 2), (2, 3)], ["float32", "float32", "float32", "int64"], "float32", {"position_ids_value": [[0, 1, 2], [3, 4, 5]], "interleaved": 0, "rotary_embedding_dim": 0}),
    (Cast, "cast", [(8, 8)], ["float32"], "int64"),
    (Cast, "cast", [(8, 8)], ["float32"], "bool"),
    (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "int64"], "int64"),
    (CastLike, "cast_like", [(8, 8), (1,)], ["float32", "bool"], "bool"),
    (BitCast, "bitcast", [(8, 8)], ["float32"], "int32"),
    (BitCast, "bitcast", [(8, 8)], ["int32"], "float32"),
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
    (Binarizer, "binarizer", [(8, 8)], ["float32"], "float32", {"threshold": 0.3, "input_values": [-1.0, -0.25, 0.0, 0.3, 0.3001, 0.75, 1.0, 2.0] * 8}),
    (HardSwish, "hard_swish", [(64, 64)], ["float32"], "float32"),
    (Swish, "swish", [(64, 64)], ["float32"], "float32", {"alpha": 1.5}),
    (Shrink, "shrink", [(64, 64)], ["float32"], "float32", {"bias": 0.2, "lambd": 0.5}),
    (Gelu, "gelu", [(64, 64)], ["float32"], "float32"),
    (Gelu, "gelu", [(64, 64)], ["float32"], "float32", {"approximate": "tanh"}),
    (Mish, "mish", [(64, 64)], ["float32"], "float32"),
    (Round, "round", [(64, 64)], ["float32"], "float32"),
    (Erf, "erf", [(64, 64)], ["float32"], "float32"),
    (Acos, "acos", [(64, 64)], ["float32"], "float32"),
    (Asin, "asin", [(64, 64)], ["float32"], "float32"),
    (Cosh, "cosh", [(64, 64)], ["float32"], "float32"),
    (Sinh, "sinh", [(64, 64)], ["float32"], "float32"),
    (Asinh, "asinh", [(64, 64)], ["float32"], "float32"),
    (Acosh, "acosh", [(64, 64)], ["float32"], "float32"),
    (Atanh, "atanh", [(64, 64)], ["float32"], "float32"),

    # 索引
    (GatherElements, "gather_elements", [(64,64), (64,64)], ["float32", "int64"], "float32", {"axis":1}),
    (GatherND, "gathernd", [(64,64), (256,2)], ["float32", "int64"], "float32"),

    # 扫描
    (CumSum, "cumsum", [(1024,)], ["float32"], "float32", {"exclusive":0, "reverse":0}),
    (CumProd, "cumprod", [(16,)], ["float32"], "float32", {"exclusive": 1, "reverse": 1}),

    (NonZero, "nonzero", [(64,64)], ["float32"], "int64"),

    (ArgMin, "argmin", [(64,64)], ["float32"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),

    (ArgMax, "argmax", [(64,64)], ["float32"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),

    # Resize: x, roi, scales, sizes
    (Resize, "resize", [(1,3,8,8), (0,), (0,), (4,)], ["float32", "float32", "float32", "int64"], "float32", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1,3,16,16]}),

    (AffineGrid, "affine_grid", [(2, 2, 3), (4,)], ["float32", "int64"], "float32", {"size_value": [2, 1, 3, 4], "align_corners": 0}),
    (Expand, "expand", [(2, 1, 3), (3,)], ["float32", "int64"], "float32", {"target_shape": [2, 4, 3]}),
    (ConstantOfShape, "constant_of_shape", [(3,)], ["int64"], "float32", {"shape_value": [2, 3, 4], "fill_value": -1.5}),
    (EyeLike, "eye_like", [(4, 5)], ["float32"], "float32", {"k": 1}),
    (Flatten, "flatten", [(2, 3, 4)], ["float32"], "float32", {"axis": -1}),
    (Squeeze, "squeeze", [(1, 2, 1, 3)], ["float32"], "float32", {"axes": [0, 2]}),
    (Unsqueeze, "unsqueeze", [(2, 3)], ["float32"], "float32", {"axes": [0, 2]}),
    (Reshape, "reshape", [(2, 3, 4), (2,)], ["float32", "int64"], "float32", {"target_shape": [0, -1]}),
    (Transpose, "transpose", [(2, 3, 4)], ["float32"], "float32", {"perm": [2, 0, 1]}),
    (Tile, "tile", [(2, 3), (2,)], ["float32", "int64"], "float32", {"repeats_value": [2, 3]}),
    (Concat, "concat", [(2, 2, 4), (2, 3, 4)], ["float32", "float32"], "float32", {"axis": 1}),
    (Pad, "pad", [(2, 3, 4), (6,), (1,)], ["float32", "int64", "float32"], "float32", {"mode": "constant", "pads_value": [0, 1, 1, 0, 1, 0], "constant_value": -2.0}),
    (CenterCropPad, "center_crop_pad", [(2, 3, 4), (3,)], ["float32", "int64"], "float32", {"target_shape": [3, 2, 5]}),
    (Slice, "slice", [(2, 4, 3), (3,), (3,), (3,), (3,)], ["float32", "int64", "int64", "int64", "int64"], "float32", {"starts_value": [0, 1, 0], "ends_value": [2, 4, 3], "axes_value": [0, 1, 2], "steps_value": [1, 2, 1]}),
    (Compress, "compress", [(2, 4, 3), (4,)], ["float32", "bool"], "float32", {"axis": 1, "condition_value": [True, False, True, False]}),
    (Compress, "compress", [(2, 3), (6,)], ["float32", "bool"], "float32", {"axis": None, "condition_value": [True, False, True, False, False, True]}),
    (ScatterElements, "scatter_elements", [(3, 4), (3, 4), (3, 4)], ["float32", "int64", "float32"], "float32", {"axis": 1, "reduction": "none"}),
    (ScatterElements, "scatter_elements", [(3, 4), (3, 4), (3, 4)], ["float32", "int64", "float32"], "float32", {"axis": 1, "reduction": "add"}),
    (ScatterElements, "scatter_elements", [(3, 4), (3, 4), (3, 4)], ["float32", "int64", "float32"], "float32", {"axis": 1, "reduction": "mul"}),
    (DepthToSpace, "depth_to_space", [(1, 8, 2, 3)], ["float32"], "float32", {"blocksize": 2, "mode": "DCR"}),
    (DepthToSpace, "depth_to_space", [(1, 8, 2, 3)], ["float32"], "float32", {"blocksize": 2, "mode": "CRD"}),
    (SpaceToDepth, "space_to_depth", [(1, 2, 4, 6)], ["float32"], "float32", {"blocksize": 2}),

    # Einsum: 当前固定主路径 ij,jk->ik
    (Einsum, "einsum", [(16,32), (32,8)], ["float32", "float32"], "float32", {"equation": "ij,jk->ik"}),

    (TopK, "topk", [(32, 64), (1,)], ["float32", "int64"], "float32",{"axis": 1, "largest": 1, "sorted": 1, "k_value": 8}),

    (RandomUniform, "random_uniform", [], [], "float32", {"shape": [32, 32], "dtype": "float32", "low": -1.0, "high": 1.0, "seed": 121.0}),
    (RandomUniformLike, "random_uniform_like", [(32, 32)], ["float32"], "float32", {"low": -1.0, "high": 1.0, "seed": 123.0}),
    (RandomNormal, "random_normal", [], [], "float32", {"shape": [32, 32], "dtype": "float32", "mean": 0.25, "scale": 0.75, "seed": 131.0}),
    (RandomNormalLike, "random_normal_like", [(32, 32)], ["float32"], "float32", {"mean": -0.5, "scale": 1.25, "seed": 137.0}),
    (Bernoulli, "bernoulli", [(4, 4)], ["float32"], "float32", {"seed": 139.0, "prob_values": [0.0, 1.0, 0.25, 0.75, 0.5, 0.1, 0.9, 0.33, 0.66, 0.2, 0.8, 0.45, 0.55, 0.05, 0.95, 0.4]}),
    (Multinomial, "multinomial", [(3, 4)], ["float32"], "int64", {"sample_size": 5, "seed": 149.0, "prob_values": [0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4]}),
    (Multinomial, "multinomial", [(2, 5)], ["float32"], "int32", {"sample_size": 4, "seed": 151.0, "prob_values": [0.5, 0.0, 1.5, 0.0, 3.0, 0.0, 4.0, 0.0, 1.0, 0.0]}),
    (NegativeLogLikelihoodLoss, "negative_log_likelihood_loss", [(2, 3, 2), (2, 2), (3,)], ["float32", "int64", "float32"], "float32", {"reduction": "mean", "ignore_index": -1, "input_values": [-0.1, -0.2, -1.0, -1.1, -2.0, -2.1, -0.3, -0.4, -1.2, -1.3, -2.2, -2.3], "target_values": [0, 2, 1, -1], "weight_values": [1.0, 2.0, 3.0]}),
    (NegativeLogLikelihoodLoss, "negative_log_likelihood_loss", [(2, 3), (2,), None], ["float16", "int64", "float16"], "float16", {"reduction": "sum", "input_values": [-0.25, -1.5, -2.0, -1.0, -0.5, -3.0], "target_values": [1, 2]}),
    (NegativeLogLikelihoodLoss, "negative_log_likelihood_loss", [(2, 3), (2,), (3,)], ["bfloat16", "int64", "bfloat16"], "bfloat16", {"reduction": "none", "ignore_index": -1, "input_values": [-0.25, -1.5, -2.0, -1.0, -0.5, -3.0], "target_values": [1, -1], "weight_values": [1.0, 2.0, 3.0]}),
    (SoftmaxCrossEntropyLoss, "softmax_cross_entropy_loss", [(2, 3), (2,), None], ["float32", "int64", "float32"], "float32", {"reduction": "none", "emit_log_prob": 1, "score_values": [1.0, 2.0, 4.0, 0.5, 0.0, -1.0], "target_values": [2, 0]}),
    (SoftmaxCrossEntropyLoss, "softmax_cross_entropy_loss", [(2, 3, 2), (2, 2), (3,)], ["float16", "int64", "float16"], "float16", {"reduction": "mean", "ignore_index": -1, "emit_log_prob": 1, "score_values": [0.5, 1.0, -0.5, 0.25, 2.0, -1.0, -0.25, 0.75, 1.5, -0.75, 0.1, 0.6], "target_values": [0, 2, 1, -1], "weight_values": [1.0, 1.5, 0.75]}),
    (SoftmaxCrossEntropyLoss, "softmax_cross_entropy_loss", [(2, 3), (2,), (3,)], ["bfloat16", "int64", "bfloat16"], "bfloat16", {"reduction": "sum", "emit_log_prob": 1, "score_values": [1.0, 2.0, 4.0, 0.5, 0.0, -1.0], "target_values": [2, 0], "weight_values": [1.0, 2.0, 0.5]}),
    (NonMaxSuppression, "non_max_suppression", [(1, 4, 4), (1, 2, 4), (1,), (1,), (1,)], ["float32", "float32", "int64", "float32", "float32"], "int64", {"center_point_box": 0, "max_output_value": 2, "iou_threshold_value": 0.5, "score_threshold_value": 0.25, "boxes_values": [0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1, 0.0, 10.0, 1.0, 11.0, 2.0, 2.0, 3.0, 3.0], "scores_values": [0.9, 0.8, 0.7, 0.2, 0.1, 0.95, 0.85, 0.4]}),
    (NonMaxSuppression, "non_max_suppression", [(1, 4, 4), (1, 1, 4), (1,), (1,), (1,)], ["bfloat16", "bfloat16", "int64", "bfloat16", "bfloat16"], "int64", {"center_point_box": 1, "max_output_value": 3, "iou_threshold_value": 0.45, "score_threshold_value": 0.5, "boxes_values": [0.5, 0.5, 1.0, 1.0, 0.55, 0.5, 1.0, 1.0, 10.5, 0.5, 1.0, 1.0, 0.5, 10.5, 1.0, 1.0], "scores_values": [0.95, 0.9, 0.8, 0.7]}),
    (NonMaxSuppression, "non_max_suppression", [(2, 5, 4), (2, 2, 5), (1,), (1,), (1,)], ["float32", "float32", "int64", "float32", "float32"], "int64", {"center_point_box": 0, "max_output_value": 2, "iou_threshold_value": 0.0, "score_threshold_value": 0.5, "boxes_values": [0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 1.0, 4.0, 0.0, 6.0, 1.0, 7.0, 3.0, 0.0, 4.0, 1.0, 3.0, 3.0, 4.0, 4.0, 10.0, 0.0, 11.0, 1.0, 10.0, 3.0, 11.0, 4.0, 10.0, 6.0, 11.0, 7.0, 13.0, 0.0, 14.0, 1.0, 13.0, 3.0, 14.0, 4.0], "scores_values": [0.9, 0.9, 0.7, 0.5, 0.1, 0.4, 0.8, 0.8, 0.2, 0.6, 0.3, 0.95, 0.95, 0.94, 0.2, 0.5, 0.5, 0.49, 0.48, 0.47]}),
    (NonMaxSuppression, "non_max_suppression", [(1, 3, 4), (1, 2, 3), (1,), (1,), (1,)], ["float16", "float16", "int64", "float16", "float16"], "int64", {"center_point_box": 0, "max_output_value": 3, "iou_threshold_value": 0.5, "score_threshold_value": 0.95, "boxes_values": [0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0], "scores_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}),
    (Dropout, "dropout", [(2, 3), (1,), (1,)], ["float32", "float32", "bool"], "float32", {"ratio_value": 0.5, "training_mode_value": 1, "seed": 0, "input_values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}),
    (DynamicQuantizeLinear, "dynamic_quantize_linear", [(2, 4)], ["float32"], "uint8", {"input_values": [-3.0, -1.25, -0.1, 0.0, 0.2, 1.7, 3.4, 6.0]}),
    (Split, "split", [(2, 6), (3,)], ["float32", "int64"], "float32", {"axis": 1, "split_value": [1, 3, 2], "num_outputs": 3, "input_values": [-3.0, -2.5, -1.0, -0.25, 0.0, 0.5, 1.25, 2.0, 3.5, 4.0, 5.25, 6.0]}),
    (Unique, "unique", [(8,)], ["int64"], "int64", {"sorted": 0, "input_values": [3, 1, 3, 2, 1, 3, -1, 2]}),
    (Unique, "unique", [(8,)], ["float32"], "float32", {"sorted": 1, "input_values": [2.0, -1.0, 2.0, 0.5, -1.0, 3.0, 0.5, 4.0]}),

    (Range, "range", [(1,), (1,), (1,)], ["float32", "float32", "float32"], "float32", {"start_value": -2.0, "limit_value": 3.0, "delta_value": 0.75}),
    (OneHot, "one_hot", [(2, 3), (), (2,)], ["int64", "int64", "float32"], "float32", {"axis": -1, "depth_value": 4, "values_value": [-0.5, 2.0]}),
    (ReverseSequence, "reverse_sequence", [(4, 3, 2), (3,)], ["float32", "int64"], "float32", {"time_axis": 0, "batch_axis": 1, "sequence_lens_value": [4, 2, 3]}),
    (Det, "det", [(2, 3, 3)], ["float32"], "float32"),
    (MelWeightMatrix, "mel_weight_matrix", [(1,), (1,), (1,), (1,), (1,)], ["int64", "int64", "int64", "float32", "float32"], "float32", {"num_mel_bins_value": 4, "dft_length_value": 10, "sample_rate_value": 16000, "lower_edge_hertz_value": 20.0, "upper_edge_hertz_value": 7600.0}),

    (Tril, "tril", [(3, 4), ()], ["float32", "int64"], "float32", {"k_value": -1}),
    (Triu, "triu", [(3, 4), ()], ["float32", "int64"], "float32", {"k_value": 1}),
    (Trilu, "trilu", [(3, 4), ()], ["float32", "int64"], "float32", {"upper": 1, "k_value": 0}),
    (HannWindow, "hann_window", [()], ["int64"], "float32", {"periodic": 1, "window_size_value": 8}),
    (HammingWindow, "hamming_window", [()], ["int64"], "float32", {"periodic": 0, "window_size_value": 9}),
    (BlackmanWindow, "blackman_window", [()], ["int64"], "float32", {"periodic": 1, "window_size_value": 10}),

    (DFT, "dft", [(1, 4, 1), ()], ["float32", "int64"], "float32", {"axis": 1, "onesided": 1, "inverse": 0, "dft_length_value": 4}),
    (DFT, "dft", [(1, 4, 2), ()], ["float32", "int64"], "float32", {"axis": 1, "onesided": 0, "inverse": 0, "dft_length_value": 4, "dft_variant": "complex_full"}),
    (DFT, "dft", [(1, 3, 2), ()], ["float32", "int64"], "float32", {"axis": 1, "onesided": 1, "inverse": 1, "dft_length_value": 4, "dft_variant": "inverse_onesided"}),
    (DFT, "dft", [(2, 3, 4, 1), ()], ["float32", "int64"], "float32", {"axis": 1, "onesided": 0, "inverse": 0, "dft_length_value": 5, "dft_variant": "high_rank_axis"}),
    (STFT, "stft", [(1, 4, 1), (), (2,), ()], ["float32", "int64", "float32", "int64"], "float32", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 2}),
    (STFT, "stft", [(1, 5, 1), (), (3,), ()], ["float32", "int64", "float32", "int64"], "float32", {"onesided": 0, "frame_step_value": 2, "frame_length_value": 3, "stft_variant": "real_window_full"}),
    (STFT, "stft", [(1, 5, 2), (), (3,), ()], ["float32", "int64", "float32", "int64"], "float32", {"onesided": 0, "frame_step_value": 2, "frame_length_value": 3, "stft_variant": "complex_no_window_full"}),
    (STFT, "stft", [(2, 2, 6, 1), (), (4,), ()], ["float32", "int64", "float32", "int64"], "float32", {"onesided": 1, "frame_step_value": 2, "frame_length_value": 4, "stft_variant": "high_rank_prefix"}),
    (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0}),
    (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "reverse", "layout": 0}),
    (RNN, "rnn", [(2, 3, 2), (2, 2, 2), (2, 2, 2), (2, 4), (2,), (2, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "bidirectional", "layout": 1}),
    (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "activations": ["Relu"], "clip": 0.35}),
    (RNN, "rnn", [(3, 2, 2), (1, 2, 2), (1, 2, 2), (1, 4), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "sequence_lens_value": [0, 2]}),
    (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 1}),
    (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 0}),
    (GRU, "gru", [(2, 3, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "reverse", "layout": 1, "linear_before_reset": 1}),
    (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 0, "activations": ["HardSigmoid", "ScaledTanh"], "activation_alpha": [0.25, 1.1], "activation_beta": [0.45, 0.7], "clip": 0.4}),
    (GRU, "gru", [(3, 2, 2), (1, 6, 2), (1, 6, 2), (1, 12), (2,), (1, 2, 2)], ["float32", "float32", "float32", "float32", "int64", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "linear_before_reset": 1, "sequence_lens_value": [0, 2]}),
    (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float32", "float32", "float32", "float32", "int64", "float32", "float32", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 1}),
    (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float32", "float32", "float32", "float32", "int64", "float32", "float32", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 0}),
    (LSTM, "lstm", [(2, 3, 2), (2, 8, 2), (2, 8, 2), (2, 16), (2,), (2, 2, 2), (2, 2, 2), (2, 6)], ["float32", "float32", "float32", "float32", "int64", "float32", "float32", "float32"], "float32", {"hidden_size": 2, "direction": "bidirectional", "layout": 1, "input_forget": 0}),
    (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float32", "float32", "float32", "float32", "int64", "float32", "float32", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 0, "activations": ["HardSigmoid", "Tanh", "Relu"], "activation_alpha": [0.25], "activation_beta": [0.45], "clip": 0.35}),
    (LSTM, "lstm", [(3, 2, 2), (1, 8, 2), (1, 8, 2), (1, 16), (2,), (1, 2, 2), (1, 2, 2), (1, 6)], ["float32", "float32", "float32", "float32", "int64", "float32", "float32", "float32"], "float32", {"hidden_size": 2, "direction": "forward", "layout": 0, "input_forget": 1, "sequence_lens_value": [0, 2]}),
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
