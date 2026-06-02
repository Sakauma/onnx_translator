# /**
#   ******************************************************************************
#   * @file        node_factories_03.py
#   * @author      Egor Izmaylov
#   * @brief       注册一组 ONNX 节点工厂，将节点属性解析为内部算子对象。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import onnx
import numpy as np
from onnx import numpy_helper

import nn.Operators
from nn import onnx_dtype_mapping

from .registry import register_factory


@register_factory("Multinomial")
def _factory_093_multinomial(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    dtype_attr, sample_size, seed = onnx.TensorProto.INT32, 1, None
    for attr in node.attribute:
        if attr.name == "dtype": dtype_attr = attr.i
        elif attr.name == "sample_size": sample_size = attr.i
        elif attr.name == "seed": seed = attr.f
    onnx_graph_list.append(nn.Operators.Multinomial(node.input, node.output, dtype=dtype_attr, sample_size=sample_size, seed=seed, version="17"))
    return onnx_graph_list[-1]


@register_factory("NegativeLogLikelihoodLoss")
def _factory_094_negativeloglikelihoodloss(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    reduction, ignore_index = "mean", None
    for attr in node.attribute:
        if attr.name == "reduction": reduction = attr.s.decode("utf-8")
        elif attr.name == "ignore_index": ignore_index = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.NegativeLogLikelihoodLoss(node.input, node.output, reduction=reduction, ignore_index=ignore_index, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("SoftmaxCrossEntropyLoss")
def _factory_095_softmaxcrossentropyloss(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    reduction, ignore_index = "mean", None
    for attr in node.attribute:
        if attr.name == "reduction": reduction = attr.s.decode("utf-8")
        elif attr.name == "ignore_index": ignore_index = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SoftmaxCrossEntropyLoss(node.input, node.output, reduction=reduction, ignore_index=ignore_index, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("MelWeightMatrix")
def _factory_096_melweightmatrix(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    output_datatype = onnx.TensorProto.FLOAT
    for attr in node.attribute:
        if attr.name == "output_datatype": output_datatype = attr.i
    onnx_graph_list.append(nn.Operators.MelWeightMatrix(node.input, node.output, output_datatype=output_datatype, version="17"))
    return onnx_graph_list[-1]


@register_factory("DFT")
def _factory_097_dft(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, inverse, onesided = 1, 0, 0
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "inverse": inverse = attr.i
        elif attr.name == "onesided": onesided = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.DFT(node.input, node.output, axis=axis, inverse=inverse, onesided=onesided, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("STFT")
def _factory_098_stft(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    onesided = 1
    for attr in node.attribute:
        if attr.name == "onesided": onesided = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.STFT(node.input, node.output, onesided=onesided, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Unique")
def _factory_099_unique(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, sorted_ = None, 1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "sorted": sorted_ = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Unique(node.input, node.output, axis=axis, sorted=sorted_, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Einsum")
def _factory_100_einsum(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    equation = ""
    for attr in node.attribute:
        if attr.name == "equation": equation = attr.s.decode('utf-8')
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Einsum(node.input, node.output, equation=equation, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Upsample")
def _factory_101_upsample(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    mode = "nearest"
    for attr in node.attribute:
        if attr.name == "mode": mode = attr.s.decode('utf-8')
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Resize(node.input, node.output, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Elu")
def _factory_102_elu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha = 1.0
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Elu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Selu")
def _factory_103_selu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha, gamma = 1.67326, 1.0507
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
        elif attr.name == "gamma": gamma = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Selu(node.input, node.output, alpha=alpha, gamma=gamma, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("LeakyRelu")
def _factory_104_leakyrelu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha = 0.01
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.LeakyRelu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("ThresholdedRelu")
def _factory_105_thresholdedrelu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha = 1.0
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ThresholdedRelu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("HardSigmoid")
def _factory_106_hardsigmoid(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha, beta = 0.2, 0.5
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
        elif attr.name == "beta": beta = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.HardSigmoid(node.input, node.output, alpha=alpha, beta=beta, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Celu")
def _factory_107_celu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha = 1.0
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Celu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Shrink")
def _factory_108_shrink(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    bias, lambd = 0.0, 0.5
    for attr in node.attribute:
        if attr.name == "bias": bias = attr.f
        elif attr.name == "lambd": lambd = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Shrink(node.input, node.output, bias=bias, lambd=lambd, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("BitwiseAnd")
@register_factory("BitwiseOr")
@register_factory("BitwiseXor")
@register_factory("BitwiseNot")
def _factory_109_bitwiseand_bitwiseor_bitwisexor(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    cls_map = {"BitwiseAnd": nn.Operators.BitwiseAnd, "BitwiseOr": nn.Operators.BitwiseOr, "BitwiseXor": nn.Operators.BitwiseXor, "BitwiseNot": nn.Operators.BitwiseNot}
    onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("BitShift")
def _factory_110_bitshift(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    direction = "LEFT"
    for attr in node.attribute:
        if attr.name == "direction": direction = attr.s.decode('utf-8')
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.BitShift(node.input, node.output, direction=direction, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("AveragePool")
def _factory_111_averagepool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    kernel_shape, pads, strides, dilations, count_include_pad, ceil_mode, auto_pad = [1,1], [0]*4, [1,1], [1,1], 0, 0, "NOTSET"
    for attr in node.attribute:
        if attr.name == "kernel_shape": kernel_shape = attr.ints
        elif attr.name == "pads": pads = attr.ints
        elif attr.name == "strides": strides = attr.ints
        elif attr.name == "dilations": dilations = attr.ints
        elif attr.name == "count_include_pad": count_include_pad = attr.i
        elif attr.name == "ceil_mode": ceil_mode = attr.i
        elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.AveragePool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations, count_include_pad=count_include_pad, ceil_mode=ceil_mode, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("LpPool")
def _factory_112_lppool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    kernel_shape, pads, strides, dilations, p, ceil_mode, auto_pad = [1,1], [0]*4, [1,1], [1,1], 2, 0, "NOTSET"
    for attr in node.attribute:
        if attr.name == "kernel_shape": kernel_shape = attr.ints
        elif attr.name == "pads": pads = attr.ints
        elif attr.name == "strides": strides = attr.ints
        elif attr.name == "dilations": dilations = attr.ints
        elif attr.name == "p": p = attr.i
        elif attr.name == "ceil_mode": ceil_mode = attr.i
        elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.LpPool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations, p=p, ceil_mode=ceil_mode, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("GlobalAveragePool")
def _factory_113_globalaveragepool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.GlobalAveragePool(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("GlobalMaxPool")
def _factory_114_globalmaxpool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.GlobalMaxPool(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("GlobalLpPool")
def _factory_115_globallppool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    p = 2
    for attr in node.attribute:
        if attr.name == "p": p = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.GlobalLpPool(node.input, node.output, p=p, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Mean")
def _factory_116_mean(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Mean(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Size")
def _factory_117_size(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    onnx_graph_list.append(nn.Operators.Size(node.input, node.output, dtype="int64", version="17"))
    return onnx_graph_list[-1]


@register_factory("IsInf")
def _factory_118_isinf(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    detect_neg, detect_pos = 1, 1
    for attr in node.attribute:
        if attr.name == "detect_negative": detect_neg = attr.i
        elif attr.name == "detect_positive": detect_pos = attr.i
    onnx_graph_list.append(nn.Operators.IsInf(node.input, node.output, detect_negative=detect_neg, detect_positive=detect_pos, version="17"))
    return onnx_graph_list[-1]


@register_factory("OneHot")
def _factory_119_onehot(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = -1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.input[2]) # Values dtype
    onnx_graph_list.append(nn.Operators.OneHot(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Tril")
def _factory_120_tril(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Tril(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Triu")
def _factory_121_triu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Triu(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Trilu")
def _factory_122_trilu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    upper = 1
    for attr in node.attribute:
        if attr.name == "upper": upper = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Trilu(node.input, node.output, upper=upper, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("BatchNormalization")
def _factory_123_batchnormalization(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    epsilon, momentum, training_mode = 1e-5, 0.9, 0
    for attr in node.attribute:
        if attr.name == "epsilon": epsilon = attr.f
        elif attr.name == "momentum": momentum = attr.f
        elif attr.name == "training_mode": training_mode = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.BatchNormalization(node.input, node.output, epsilon=epsilon, momentum=momentum, training_mode=training_mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("InstanceNormalization")
def _factory_124_instancenormalization(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    epsilon = 1e-5
    for attr in node.attribute:
        if attr.name == "epsilon": epsilon = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.InstanceNormalization(node.input, node.output, epsilon=epsilon, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("LayerNormalization")
def _factory_125_layernormalization(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    epsilon, axis, stash_type = 1e-5, -1, 1
    for attr in node.attribute:
        if attr.name == "epsilon": epsilon = attr.f
        elif attr.name == "axis": axis = attr.i
        elif attr.name == "stash_type": stash_type = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.LayerNormalization(node.input, node.output, axis=axis, epsilon=epsilon, stash_type=stash_type, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("HannWindow")
@register_factory("HammingWindow")
@register_factory("BlackmanWindow")
def _factory_126_hannwindow_hammingwindow_blackmanwindow(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    periodic, output_datatype = 1, 1
    for attr in node.attribute:
        if attr.name == "periodic": periodic = attr.i
        elif attr.name == "output_datatype": output_datatype = attr.i
    cls_map = {"HannWindow": nn.Operators.HannWindow, "HammingWindow": nn.Operators.HammingWindow, "BlackmanWindow": nn.Operators.BlackmanWindow}
    onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, periodic=periodic, output_datatype=output_datatype, version="17"))
    return onnx_graph_list[-1]


@register_factory("RandomNormal")
def _factory_127_randomnormal(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    mean, scale, seed, dtype, shape = 0.0, 1.0, 0.0, 1, []
    for attr in node.attribute:
        if attr.name == "mean": mean = attr.f
        elif attr.name == "scale": scale = attr.f
        elif attr.name == "seed": seed = attr.f
        elif attr.name == "dtype": dtype = attr.i
        elif attr.name == "shape": shape = attr.ints
    onnx_graph_list.append(nn.Operators.RandomNormal(node.input, node.output, mean=mean, scale=scale, seed=seed, dtype=dtype, shape=shape, version="17"))
    return onnx_graph_list[-1]


@register_factory("RandomNormalLike")
def _factory_128_randomnormallike(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    mean, scale, seed, dtype = 0.0, 1.0, 0.0, None
    for attr in node.attribute:
        if attr.name == "mean": mean = attr.f
        elif attr.name == "scale": scale = attr.f
        elif attr.name == "seed": seed = attr.f
        elif attr.name == "dtype": dtype = attr.i
    onnx_graph_list.append(nn.Operators.RandomNormalLike(node.input, node.output, mean=mean, scale=scale, seed=seed, dtype=dtype, version="17"))
    return onnx_graph_list[-1]


@register_factory("Bernoulli")
def _factory_129_bernoulli(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    seed, dtype = 0.0, None
    for attr in node.attribute:
        if attr.name == "seed": seed = attr.f
        elif attr.name == "dtype": dtype = attr.i
    onnx_graph_list.append(nn.Operators.Bernoulli(node.input, node.output, seed=seed, dtype=dtype, version="17"))
    return onnx_graph_list[-1]


@register_factory("Dropout")
def _factory_130_dropout(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    seed, ratio, training_mode = None, 0.5, 0
    for attr in node.attribute:
        if attr.name == "seed": seed = attr.i if attr.type == onnx.AttributeProto.INT else attr.f
        elif attr.name == "ratio": ratio = attr.f
        elif attr.name == "training_mode": training_mode = attr.i
    onnx_graph_list.append(nn.Operators.Dropout(node.input, node.output, seed=seed, ratio=ratio, training_mode=training_mode, version="17"))
    return onnx_graph_list[-1]


@register_factory("Hardmax")
def _factory_131_hardmax(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = -1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Hardmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("LogSoftmax")
def _factory_132_logsoftmax(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = -1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.LogSoftmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("LpNormalization")
def _factory_133_lpnormalization(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, p = -1, 2
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "p": p = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.LpNormalization(node.input, node.output, axis=axis, p=p, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("DepthToSpace")
def _factory_134_depthtospace(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    blocksize, mode = 1, "DCR"
    for attr in node.attribute:
        if attr.name == "blocksize": blocksize = attr.i
        elif attr.name == "mode": mode = attr.s.decode('utf-8')
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.DepthToSpace(node.input, node.output, blocksize=blocksize, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("SpaceToDepth")
def _factory_135_spacetodepth(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    blocksize = 1
    for attr in node.attribute:
        if attr.name == "blocksize": blocksize = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SpaceToDepth(node.input, node.output, blocksize=blocksize, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("ReverseSequence")
def _factory_136_reversesequence(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    time_axis, batch_axis = 0, 1
    for attr in node.attribute:
        if attr.name == "time_axis": time_axis = attr.i
        elif attr.name == "batch_axis": batch_axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ReverseSequence(node.input, node.output, time_axis=time_axis, batch_axis=batch_axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Compress")
def _factory_137_compress(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = None
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Compress(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("ScatterElements")
@register_factory("Scatter")
def _factory_138_scatterelements_scatter(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, reduction = 0, "none"
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "reduction": reduction = attr.s.decode('utf-8')
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ScatterElements(node.input, node.output, axis=axis, reduction=reduction, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("GroupNormalization")
def _factory_139_groupnormalization(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    num_groups, epsilon = 1, 1e-5
    for attr in node.attribute:
        if attr.name == "num_groups": num_groups = attr.i
        elif attr.name == "epsilon": epsilon = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.GroupNormalization(node.input, node.output, num_groups=num_groups, epsilon=epsilon, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("StringNormalizer")
def _factory_140_stringnormalizer(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    case_change_action, is_case_sensitive, locale, stopwords = "NONE", 0, "", []
    for attr in node.attribute:
        if attr.name == "case_change_action": case_change_action = attr.s.decode("utf-8")
        elif attr.name == "is_case_sensitive": is_case_sensitive = attr.i
        elif attr.name == "locale": locale = attr.s.decode("utf-8")
        elif attr.name == "stopwords": stopwords = [item.decode("utf-8") for item in attr.strings]
    onnx_graph_list.append(nn.Operators.StringNormalizer(
        node.input,
        node.output,
        case_change_action=case_change_action,
        is_case_sensitive=is_case_sensitive,
        locale=locale,
        stopwords=stopwords,
        version="17",
    ))
    return onnx_graph_list[-1]


