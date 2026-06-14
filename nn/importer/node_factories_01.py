# /**
#   ******************************************************************************
#   * @file        node_factories_01.py
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


@register_factory("RELU")
def _factory_001_relu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.RELU(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("COS")
def _factory_002_cos(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.COS(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("ABS")
def _factory_003_abs(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ABS(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("ADD")
def _factory_004_add(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ADD(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("SUB")
def _factory_005_sub(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SUB(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("MUL")
def _factory_006_mul(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.MUL(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("DIV")
def _factory_007_div(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.DIV(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Conv")
def _factory_008_conv(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    pads, strides, dilations, group, kernel_shape, auto_pad = None, None, None, 1, None, "NOTSET"
    for attr in node.attribute:
        if attr.name == "pads": pads = list(attr.ints)
        elif attr.name == "strides": strides = list(attr.ints)
        elif attr.name == "dilations": dilations = list(attr.ints)
        elif attr.name == "group": group = attr.i
        elif attr.name == "kernel_shape": kernel_shape = list(attr.ints)
        elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Conv(node.input, node.output, pads=pads, strides=strides, dilations=dilations, group=group, kernel_shape=kernel_shape, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("ConvTranspose")
def _factory_009_convtranspose(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    attrs = {
        "pads": None,
        "strides": None,
        "dilations": None,
        "group": 1,
        "kernel_shape": None,
        "output_padding": None,
        "output_shape": None,
        "auto_pad": "NOTSET",
    }
    for attr in node.attribute:
        if attr.name in {"pads", "strides", "dilations", "kernel_shape", "output_padding", "output_shape"}:
            attrs[attr.name] = list(attr.ints)
        elif attr.name == "group":
            attrs["group"] = attr.i
        elif attr.name == "auto_pad":
            attrs["auto_pad"] = attr.s.decode("utf-8")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ConvTranspose(
        node.input,
        node.output,
        pads=attrs["pads"],
        strides=attrs["strides"],
        dilations=attrs["dilations"],
        group=attrs["group"],
        kernel_shape=attrs["kernel_shape"],
        output_padding=attrs["output_padding"],
        output_shape=attrs["output_shape"],
        auto_pad=attrs["auto_pad"],
        dtype=onnx_dtype_mapping[elem_type],
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("Col2Im")
def _factory_010_col2im(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    pads, strides, dilations = None, None, None
    for attr in node.attribute:
        if attr.name == "pads":
            pads = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "dilations":
            dilations = list(attr.ints)
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(
        nn.Operators.Col2Im(
            node.input,
            node.output,
            pads=pads,
            strides=strides,
            dilations=dilations,
            dtype=onnx_dtype_mapping[elem_type],
            version="18",
        )
    )
    return onnx_graph_list[-1]


@register_factory("DeformConv")
def _factory_011_deformconv(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    attrs = {
        "strides": None,
        "pads": None,
        "dilations": None,
        "group": 1,
        "kernel_shape": None,
        "offset_group": 1,
    }
    for attr in node.attribute:
        if attr.name in {"strides", "pads", "dilations", "kernel_shape"}:
            attrs[attr.name] = list(attr.ints)
        elif attr.name == "group":
            attrs["group"] = attr.i
        elif attr.name == "offset_group":
            attrs["offset_group"] = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(
        nn.Operators.DeformConv(
            node.input,
            node.output,
            strides=attrs["strides"],
            pads=attrs["pads"],
            dilations=attrs["dilations"],
            group=attrs["group"],
            kernel_shape=attrs["kernel_shape"],
            offset_group=attrs["offset_group"],
            dtype=onnx_dtype_mapping[elem_type],
            version="22",
        )
    )
    return onnx_graph_list[-1]


@register_factory("ConvInteger")
def _factory_012_convinteger(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    pads, strides, dilations, group, kernel_shape, auto_pad = None, None, None, 1, None, "NOTSET"
    for attr in node.attribute:
        if attr.name == "pads": pads = list(attr.ints)
        elif attr.name == "strides": strides = list(attr.ints)
        elif attr.name == "dilations": dilations = list(attr.ints)
        elif attr.name == "group": group = attr.i
        elif attr.name == "kernel_shape": kernel_shape = list(attr.ints)
        elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
    onnx_graph_list.append(nn.Operators.ConvInteger(
        node.input,
        node.output,
        pads=pads,
        strides=strides,
        dilations=dilations,
        group=group,
        kernel_shape=kernel_shape,
        auto_pad=auto_pad,
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("QLinearConv")
def _factory_011_qlinearconv(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    pads, strides, dilations, group, kernel_shape, auto_pad = None, None, None, 1, None, "NOTSET"
    for attr in node.attribute:
        if attr.name == "pads": pads = list(attr.ints)
        elif attr.name == "strides": strides = list(attr.ints)
        elif attr.name == "dilations": dilations = list(attr.ints)
        elif attr.name == "group": group = attr.i
        elif attr.name == "kernel_shape": kernel_shape = list(attr.ints)
        elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.QLinearConv(
        node.input,
        node.output,
        pads=pads,
        strides=strides,
        dilations=dilations,
        group=group,
        kernel_shape=kernel_shape,
        auto_pad=auto_pad,
        dtype=onnx_dtype_mapping[elem_type],
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("RNN")
@register_factory("GRU")
@register_factory("LSTM")
def _factory_012_rnn_gru_lstm(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    hidden_size, direction, clip, layout = None, "forward", None, 0
    activations, activation_alpha, activation_beta = None, None, None
    linear_before_reset, input_forget = 0, 0
    for attr in node.attribute:
        if attr.name == "hidden_size": hidden_size = attr.i
        elif attr.name == "direction": direction = attr.s.decode("utf-8")
        elif attr.name == "clip": clip = attr.f
        elif attr.name == "layout": layout = attr.i
        elif attr.name == "activations": activations = [item.decode("utf-8") for item in attr.strings]
        elif attr.name == "activation_alpha": activation_alpha = list(attr.floats)
        elif attr.name == "activation_beta": activation_beta = list(attr.floats)
        elif attr.name == "linear_before_reset": linear_before_reset = attr.i
        elif attr.name == "input_forget": input_forget = attr.i
    out_name = next((name for name in node.output if name), node.input[0])
    elem_type = get_dtype(out_name)
    common = dict(
        hidden_size=hidden_size,
        direction=direction,
        activations=activations,
        activation_alpha=activation_alpha,
        activation_beta=activation_beta,
        clip=clip,
        layout=layout,
        dtype=onnx_dtype_mapping[elem_type],
        version="17",
    )
    if node.op_type == "RNN":
        onnx_graph_list.append(nn.Operators.RNN(node.input, node.output, **common))
    elif node.op_type == "GRU":
        onnx_graph_list.append(nn.Operators.GRU(node.input, node.output, linear_before_reset=linear_before_reset, **common))
    else:
        onnx_graph_list.append(nn.Operators.LSTM(node.input, node.output, input_forget=input_forget, **common))
    return onnx_graph_list[-1]


@register_factory("MaxPool")
def _factory_013_maxpool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    kernel_shape, pads, strides, dilations, auto_pad = [1,1], [0]*4, [1,1], [1,1], "NOTSET"
    ceil_mode, storage_order = 0, 0
    for attr in node.attribute:
        if attr.name == "kernel_shape": kernel_shape = attr.ints
        elif attr.name == "pads": pads = attr.ints
        elif attr.name == "strides": strides = attr.ints
        elif attr.name == "dilations": dilations = attr.ints
        elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
        elif attr.name == "ceil_mode": ceil_mode = attr.i
        elif attr.name == "storage_order": storage_order = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.MaxPool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations, ceil_mode=ceil_mode, storage_order=storage_order, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("MaxUnpool")
def _factory_014_maxunpool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    kernel_shape, pads, strides = None, None, None
    for attr in node.attribute:
        if attr.name == "kernel_shape": kernel_shape = list(attr.ints)
        elif attr.name == "pads": pads = list(attr.ints)
        elif attr.name == "strides": strides = list(attr.ints)
    if kernel_shape is None:
        raise ValueError("MaxUnpool requires kernel_shape attribute")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.MaxUnpool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("MaxRoiPool")
def _factory_015_maxroipool(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    pooled_shape, spatial_scale = None, 1.0
    for attr in node.attribute:
        if attr.name == "pooled_shape": pooled_shape = list(attr.ints)
        elif attr.name == "spatial_scale": spatial_scale = attr.f
    if pooled_shape is None:
        raise ValueError("MaxRoiPool requires pooled_shape attribute")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.MaxRoiPool(node.input, node.output, pooled_shape=pooled_shape, spatial_scale=spatial_scale, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("RoiAlign")
def _factory_016_roialign(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    output_height, output_width, sampling_ratio = 1, 1, 0
    spatial_scale, mode, coord_mode = 1.0, "avg", "half_pixel"
    for attr in node.attribute:
        if attr.name == "output_height": output_height = attr.i
        elif attr.name == "output_width": output_width = attr.i
        elif attr.name == "sampling_ratio": sampling_ratio = attr.i
        elif attr.name == "spatial_scale": spatial_scale = attr.f
        elif attr.name == "mode": mode = attr.s.decode("utf-8")
        elif attr.name == "coordinate_transformation_mode": coord_mode = attr.s.decode("utf-8")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.RoiAlign(
        node.input,
        node.output,
        output_height=output_height,
        output_width=output_width,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        mode=mode,
        coordinate_transformation_mode=coord_mode,
        dtype=onnx_dtype_mapping[elem_type],
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("Gemm")
def _factory_017_gemm(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha, beta, transA, transB = 1.0, 1.0, 0, 0
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
        elif attr.name == "beta": beta = attr.f
        elif attr.name == "transA": transA = attr.i
        elif attr.name == "transB": transB = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Gemm(node.input, node.output, alpha=alpha, beta=beta, transA=transA, transB=transB, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Softmax")
def _factory_018_softmax(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = -1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Softmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("QuantizeLinear")
def _factory_019_quantizelinear(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = 1
    output_dtype_proto = 0
    block_size = 0
    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i
        elif attr.name == "output_dtype":
            output_dtype_proto = attr.i
        elif attr.name == "block_size":
            block_size = attr.i
    output_dtype = onnx_dtype_mapping.get(output_dtype_proto) if output_dtype_proto else None
    if len(node.input) >= 3 and node.input[2]:
        zp_name = node.input[2]
        if zp_name in import_context.dtype_map:
            target_dtype = onnx_dtype_mapping[import_context.dtype_map[zp_name]]
        else:
            raise ValueError(f"Unknown dtype for ZeroPoint {zp_name}")
    elif output_dtype is not None:
        target_dtype = output_dtype
    else:
        try:
            target_dtype = onnx_dtype_mapping[get_dtype(node.output[0])]
        except Exception:
            target_dtype = "uint8"
    onnx_graph_list.append(nn.Operators.QuantizeLinear(node.input, node.output, axis=axis, dtype=target_dtype, output_dtype=output_dtype, block_size=block_size, version="25" if output_dtype is not None or block_size else "17"))
    return onnx_graph_list[-1]


@register_factory("DequantizeLinear")
def _factory_020_dequantizelinear(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = 1
    output_dtype_proto = 0
    block_size = 0
    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i
        elif attr.name == "output_dtype":
            output_dtype_proto = attr.i
        elif attr.name == "block_size":
            block_size = attr.i
    output_dtype = onnx_dtype_mapping.get(output_dtype_proto) if output_dtype_proto else None
    elem_type = get_dtype(node.output[0])
    target_dtype = output_dtype or onnx_dtype_mapping[elem_type]
    onnx_graph_list.append(nn.Operators.DequantizeLinear(node.input, node.output, axis=axis, dtype=target_dtype, output_dtype=output_dtype, block_size=block_size, version="25" if output_dtype is not None or block_size else "17"))
    return onnx_graph_list[-1]


@register_factory("EXP")
def _factory_021_exp(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.EXP(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("LOG")
def _factory_022_log(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.LOG(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("SQRT")
def _factory_023_sqrt(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SQRT(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("SIGMOID")
def _factory_024_sigmoid(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SIGMOID(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("TANH")
def _factory_025_tanh(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.TANH(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Flatten")
def _factory_026_flatten(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = 1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Flatten(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Reshape")
def _factory_027_reshape(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    allowzero = 0
    for attr in node.attribute:
        if attr.name == "allowzero": allowzero = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Reshape(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17", allowzero=allowzero))
    return onnx_graph_list[-1]


@register_factory("Transpose")
def _factory_028_transpose(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    perm = []
    for attr in node.attribute:
        if attr.name == "perm": perm = attr.ints
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Transpose(node.input, node.output, perm=perm, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Pow")
def _factory_029_pow(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Pow(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Max")
def _factory_030_max(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Max(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Min")
def _factory_031_min(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Min(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Squeeze")
def _factory_032_squeeze(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axes = None
    for attr in node.attribute:
        if attr.name == "axes": axes = attr.ints
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Squeeze(node.input, node.output, axes=axes, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Unsqueeze")
def _factory_033_unsqueeze(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axes = None
    for attr in node.attribute:
        if attr.name == "axes": axes = attr.ints
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Unsqueeze(node.input, node.output, axes=axes, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Concat")
def _factory_034_concat(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = 1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Concat(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Slice")
def _factory_035_slice(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Slice(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Neg")
def _factory_036_neg(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Neg(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Reciprocal")
def _factory_037_reciprocal(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Reciprocal(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Ceil")
def _factory_038_ceil(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Ceil(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Floor")
def _factory_039_floor(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Floor(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Cast")
def _factory_040_cast(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    to = 1
    for attr in node.attribute:
        if attr.name == "to": to = attr.i
    target_dtype = onnx_dtype_mapping.get(to, "float32")
    onnx_graph_list.append(nn.Operators.Cast(node.input, node.output, dtype=target_dtype, version="17"))
    return onnx_graph_list[-1]


@register_factory("CastLike")
def _factory_041_castlike(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.CastLike(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("BitCast")
def _factory_041_bitcast(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    to = None
    for attr in node.attribute:
        if attr.name == "to":
            to = attr.i
    if to is None:
        raise ValueError("BitCast requires 'to' attribute")
    if to not in onnx_dtype_mapping:
        raise ValueError(f"BitCast target TensorProto dtype {to} is not supported by current dtype mapping")
    onnx_graph_list.append(nn.Operators.BitCast(node.input, node.output, dtype=onnx_dtype_mapping[to], version="26"))
    return onnx_graph_list[-1]


@register_factory("Sum")
def _factory_042_sum(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Sum(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("PRelu")
def _factory_043_prelu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.PRelu(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Det")
def _factory_044_det(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Det(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("MatMulInteger")
def _factory_045_matmulinteger(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    onnx_graph_list.append(nn.Operators.MatMulInteger(node.input, node.output, dtype="int32", version="17"))
    return onnx_graph_list[-1]


@register_factory("QLinearMatMul")
def _factory_046_qlinearmatmul(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.QLinearMatMul(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("LRN")
def _factory_047_lrn(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    alpha, beta, bias, size = 0.0001, 0.75, 1.0, None
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
        elif attr.name == "beta": beta = attr.f
        elif attr.name == "bias": bias = attr.f
        elif attr.name == "size": size = attr.i
    if size is None:
        raise ValueError("LRN requires size attribute")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.LRN(node.input, node.output, size=size, alpha=alpha, beta=beta, bias=bias, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]
