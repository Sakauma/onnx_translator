# /**
#   ******************************************************************************
#   * @file        node_factories_02.py
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


@register_factory("MeanVarianceNormalization")
def _factory_048_meanvariancenormalization(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axes = [0, 2, 3]
    for attr in node.attribute:
        if attr.name == "axes": axes = list(attr.ints)
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.MeanVarianceNormalization(node.input, node.output, axes=axes, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("EyeLike")
def _factory_049_eyelike(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    k, dtype = 0, None
    for attr in node.attribute:
        if attr.name == "k": k = attr.i
        elif attr.name == "dtype": dtype = attr.i
    elem_type = dtype if dtype is not None else get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.EyeLike(node.input, node.output, k=k, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Clip")
def _factory_050_clip(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Clip(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("MatMul")
def _factory_051_matmul(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.MatMul(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Gather")
def _factory_052_gather(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = 0
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Gather(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Expand")
def _factory_053_expand(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Expand(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Shape")
def _factory_054_shape(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    start, end = 0, None
    for attr in node.attribute:
        if attr.name == "start": start = attr.i
        elif attr.name == "end": end = attr.i
    onnx_graph_list.append(nn.Operators.Shape(node.input, node.output, start=start, end=end, dtype="int64", version="17"))
    return onnx_graph_list[-1]


@register_factory("Constant")
def _factory_055_constant(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    value, dtype = None, "float32"
    for attr in node.attribute:
        if attr.name == "value":
            value = numpy_helper.to_array(attr.t)
            dtype = onnx_dtype_mapping[attr.t.data_type]
        elif attr.name == "sparse_value":
            value = numpy_helper.to_array(attr.sparse_tensor)
            dtype = onnx_dtype_mapping[attr.sparse_tensor.values.data_type]
        elif attr.name == "value_float":
            value = np.asarray(attr.f, dtype=np.float32)
            dtype = "float32"
        elif attr.name == "value_floats":
            value = np.asarray(list(attr.floats), dtype=np.float32)
            dtype = "float32"
        elif attr.name == "value_int":
            value = np.asarray(attr.i, dtype=np.int64)
            dtype = "int64"
        elif attr.name == "value_ints":
            value = np.asarray(list(attr.ints), dtype=np.int64)
            dtype = "int64"
        elif attr.name == "value_string":
            value = np.asarray(attr.s.decode("utf-8", errors="ignore"), dtype=np.str_)
            dtype = "string"
        elif attr.name == "value_strings":
            value = np.asarray([item.decode("utf-8", errors="ignore") for item in attr.strings], dtype=np.str_)
            dtype = "string"
    onnx_graph_list.append(nn.Operators.Constant(node.input, node.output, value=value, dtype=dtype, version="17"))
    return onnx_graph_list[-1]


@register_factory("Equal")
@register_factory("Greater")
@register_factory("Less")
@register_factory("GreaterOrEqual")
@register_factory("LessOrEqual")
@register_factory("Not")
@register_factory("And")
@register_factory("Or")
@register_factory("Xor")
@register_factory("IsNaN")
def _factory_056_equal_greater_less(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    cls_map = {
        "Equal": nn.Operators.Equal, "Greater": nn.Operators.Greater, "Less": nn.Operators.Less,
        "GreaterOrEqual": nn.Operators.GreaterOrEqual, "LessOrEqual": nn.Operators.LessOrEqual,
        "Not": nn.Operators.Not, "And": nn.Operators.And, "Or": nn.Operators.Or, "Xor": nn.Operators.Xor, "IsNaN": nn.Operators.IsNaN
    }
    onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, dtype="bool", version="17"))
    return onnx_graph_list[-1]


@register_factory("Sin")
@register_factory("Tan")
@register_factory("Atan")
@register_factory("Sign")
@register_factory("Identity")
@register_factory("Round")
@register_factory("Erf")
@register_factory("Softplus")
@register_factory("Softsign")
@register_factory("HardSwish")
@register_factory("Swish")
@register_factory("Acos")
@register_factory("Asin")
@register_factory("Cosh")
@register_factory("Sinh")
@register_factory("Asinh")
@register_factory("Acosh")
@register_factory("Atanh")
@register_factory("Mish")
def _factory_057_sin_tan_atan(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    cls_map = {
        "Sin": nn.Operators.Sin, "Tan": nn.Operators.Tan, "Atan": nn.Operators.Atan, "Sign": nn.Operators.Sign, "Identity": nn.Operators.Identity,
        "Round": nn.Operators.Round, "Erf": nn.Operators.Erf, "Softplus": nn.Operators.Softplus, "Softsign": nn.Operators.Softsign, "HardSwish": nn.Operators.HardSwish,
        "Swish": nn.Operators.Swish, "Acos": nn.Operators.Acos, "Asin": nn.Operators.Asin, "Cosh": nn.Operators.Cosh, "Sinh": nn.Operators.Sinh, "Asinh": nn.Operators.Asinh, "Acosh": nn.Operators.Acosh, "Atanh": nn.Operators.Atanh,
        "Gelu": nn.Operators.Gelu, "Mish": nn.Operators.Mish
    }
    if node.op_type == "Swish":
        alpha = 1.0
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
        onnx_graph_list.append(
            cls_map[node.op_type](
                node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="24"
            )
        )
    else:
        onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Gelu")
def _factory_058_gelu(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    approximate = "none"
    for attr in node.attribute:
        if attr.name == "approximate":
            approximate = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else str(attr.s)
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(
        nn.Operators.Gelu(
            node.input,
            node.output,
            approximate=approximate,
            dtype=onnx_dtype_mapping[elem_type],
            version="20",
        )
    )
    return onnx_graph_list[-1]


@register_factory("Mod")
def _factory_058_mod(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    fmod = 0
    for attr in node.attribute:
        if attr.name == "fmod": fmod = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Mod(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], fmod=fmod, version="17"))
    return onnx_graph_list[-1]


@register_factory("Where")
def _factory_059_where(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Where(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("ConstantOfShape")
def _factory_060_constantofshape(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    value, target_dtype = None, "float32"
    for attr in node.attribute:
        if attr.name == "value": value = numpy_helper.to_array(attr.t)
    if value is not None:
        target_dtype = nn.NUMPY_TO_DTYPE.get(value.dtype.type, target_dtype)
    onnx_graph_list.append(nn.Operators.ConstantOfShape(node.input, node.output, value=value, dtype=target_dtype, version="17"))
    return onnx_graph_list[-1]


@register_factory("Range")
def _factory_061_range(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.input[0])
    onnx_graph_list.append(nn.Operators.Range(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Tile")
def _factory_062_tile(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Tile(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Pad")
def _factory_063_pad(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    mode = "constant"
    for attr in node.attribute:
        if attr.name == "mode": mode = attr.s.decode('utf-8')
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Pad(node.input, node.output, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Split")
def _factory_064_split(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = 0
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Split(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("SequenceEmpty")
def _factory_065_sequenceempty(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    dtype = 1
    for attr in node.attribute:
        if attr.name == "dtype": dtype = attr.i
    onnx_graph_list.append(nn.Operators.SequenceEmpty(node.input, node.output, dtype=onnx_dtype_mapping.get(dtype, "float32"), version="17"))
    return onnx_graph_list[-1]


@register_factory("SequenceConstruct")
def _factory_066_sequenceconstruct(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.input[0]) if node.input else get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SequenceConstruct(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
    return onnx_graph_list[-1]


@register_factory("SequenceAt")
def _factory_067_sequenceat(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SequenceAt(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
    return onnx_graph_list[-1]


@register_factory("SequenceInsert")
def _factory_068_sequenceinsert(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.input[1]) if len(node.input) > 1 else get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SequenceInsert(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
    return onnx_graph_list[-1]


@register_factory("SequenceErase")
def _factory_069_sequenceerase(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.SequenceErase(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
    return onnx_graph_list[-1]


@register_factory("SequenceLength")
def _factory_070_sequencelength(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    onnx_graph_list.append(nn.Operators.SequenceLength(node.input, node.output, version="17"))
    return onnx_graph_list[-1]


@register_factory("ConcatFromSequence")
def _factory_071_concatfromsequence(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, new_axis = 0, 0
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "new_axis": new_axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ConcatFromSequence(node.input, node.output, axis=axis, new_axis=new_axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("SplitToSequence")
def _factory_072_splittosequence(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, keepdims = 0, 1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "keepdims": keepdims = attr.i
    elem_type = get_dtype(node.input[0])
    onnx_graph_list.append(nn.Operators.SplitToSequence(node.input, node.output, axis=axis, keepdims=keepdims, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("Optional")
def _factory_073_optional(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.input[0]) if node.input else get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Optional(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
    return onnx_graph_list[-1]


@register_factory("OptionalGetElement")
def _factory_074_optionalgetelement(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.OptionalGetElement(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
    return onnx_graph_list[-1]


@register_factory("OptionalHasElement")
def _factory_075_optionalhaselement(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    onnx_graph_list.append(nn.Operators.OptionalHasElement(node.input, node.output, version="17"))
    return onnx_graph_list[-1]


@register_factory("If")
def _factory_076_if(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    then_branch, else_branch = None, None
    for attr in node.attribute:
        if attr.name == "then_branch": then_branch = attr.g
        elif attr.name == "else_branch": else_branch = attr.g
    if then_branch is None or else_branch is None:
        raise ValueError("If requires then_branch and else_branch graphs")
    onnx_graph_list.append(nn.Operators.If(node.input, node.output, then_branch=then_branch, else_branch=else_branch, version="17"))
    return onnx_graph_list[-1]


@register_factory("Loop")
def _factory_077_loop(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    body = None
    for attr in node.attribute:
        if attr.name == "body": body = attr.g
    if body is None:
        raise ValueError("Loop requires body graph")
    onnx_graph_list.append(nn.Operators.Loop(node.input, node.output, body=body, version="17"))
    return onnx_graph_list[-1]


@register_factory("Scan")
def _factory_078_scan(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    body, num_scan_inputs = None, None
    scan_input_axes = scan_input_directions = scan_output_axes = scan_output_directions = None
    for attr in node.attribute:
        if attr.name == "body": body = attr.g
        elif attr.name == "num_scan_inputs": num_scan_inputs = attr.i
        elif attr.name == "scan_input_axes": scan_input_axes = list(attr.ints)
        elif attr.name == "scan_input_directions": scan_input_directions = list(attr.ints)
        elif attr.name == "scan_output_axes": scan_output_axes = list(attr.ints)
        elif attr.name == "scan_output_directions": scan_output_directions = list(attr.ints)
    if body is None or num_scan_inputs is None:
        raise ValueError("Scan requires body graph and num_scan_inputs")
    onnx_graph_list.append(nn.Operators.Scan(
        node.input,
        node.output,
        body=body,
        num_scan_inputs=num_scan_inputs,
        scan_input_axes=scan_input_axes,
        scan_input_directions=scan_input_directions,
        scan_output_axes=scan_output_axes,
        scan_output_directions=scan_output_directions,
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("SequenceMap")
def _factory_079_sequencemap(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    body = None
    for attr in node.attribute:
        if attr.name == "body": body = attr.g
    if body is None:
        raise ValueError("SequenceMap requires body graph")
    onnx_graph_list.append(nn.Operators.SequenceMap(node.input, node.output, body=body, version="17"))
    return onnx_graph_list[-1]


@register_factory("ReduceMean")
@register_factory("ReduceSum")
@register_factory("ReduceMax")
@register_factory("ReduceMin")
@register_factory("ReduceProd")
@register_factory("ReduceL1")
@register_factory("ReduceL2")
@register_factory("ReduceLogSum")
@register_factory("ReduceLogSumExp")
@register_factory("ReduceSumSquare")
def _factory_080_reducemean_reducesum_reducemax(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axes, keepdims, noop_with_empty_axes = None, 1, 0
    for attr in node.attribute:
        if attr.name == "axes": axes = attr.ints
        elif attr.name == "keepdims": keepdims = attr.i
        elif attr.name == "noop_with_empty_axes": noop_with_empty_axes = attr.i
    elem_type = get_dtype(node.output[0])
    cls_map = {
        "ReduceMean": nn.Operators.ReduceMean, "ReduceSum": nn.Operators.ReduceSum, "ReduceMax": nn.Operators.ReduceMax, "ReduceMin": nn.Operators.ReduceMin, "ReduceProd": nn.Operators.ReduceProd,
        "ReduceL1": nn.Operators.ReduceL1, "ReduceL2": nn.Operators.ReduceL2, "ReduceLogSum": nn.Operators.ReduceLogSum, "ReduceLogSumExp": nn.Operators.ReduceLogSumExp, "ReduceSumSquare": nn.Operators.ReduceSumSquare
    }
    kwargs = {}
    if node.op_type == "ReduceSum":
        kwargs["noop_with_empty_axes"] = noop_with_empty_axes
    onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, axes=axes, keepdims=keepdims, dtype=onnx_dtype_mapping[elem_type], version="17", **kwargs))
    return onnx_graph_list[-1]


@register_factory("ArgMax")
@register_factory("ArgMin")
def _factory_081_argmax_argmin(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, keepdims, select_last_index = 0, 1, 0
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "keepdims": keepdims = attr.i
        elif attr.name == "select_last_index": select_last_index = attr.i
    cls_map = {"ArgMax": nn.Operators.ArgMax, "ArgMin": nn.Operators.ArgMin}
    onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, axis=axis, keepdims=keepdims, select_last_index=select_last_index, dtype="int64", version="17"))
    return onnx_graph_list[-1]


@register_factory("ScatterND")
def _factory_082_scatternd(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    reduction = "none"
    for attr in node.attribute:
        if attr.name == "reduction": reduction = attr.s.decode('utf-8')
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.ScatterND(node.input, node.output, reduction=reduction, dtype=onnx_dtype_mapping[elem_type]))
    return onnx_graph_list[-1]


@register_factory("GatherND")
def _factory_083_gathernd(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    batch_dims = 0
    for attr in node.attribute:
        if attr.name == "batch_dims": batch_dims = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.GatherND(node.input, node.output, batch_dims=batch_dims, dtype=onnx_dtype_mapping[elem_type]))
    return onnx_graph_list[-1]


@register_factory("GatherElements")
def _factory_084_gatherelements(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis = 0
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.GatherElements(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type]))
    return onnx_graph_list[-1]


@register_factory("NonZero")
def _factory_085_nonzero(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    onnx_graph_list.append(nn.Operators.NonZero(node.input, node.output))
    return onnx_graph_list[-1]


@register_factory("NonMaxSuppression")
def _factory_086_nonmaxsuppression(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    center_point_box = 0
    for attr in node.attribute:
        if attr.name == "center_point_box": center_point_box = attr.i
    onnx_graph_list.append(nn.Operators.NonMaxSuppression(node.input, node.output, center_point_box=center_point_box, version="17"))
    return onnx_graph_list[-1]


@register_factory("Resize")
def _factory_087_resize(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    mode, coord_mode, nearest_mode = "nearest", "asymmetric", "round_prefer_floor"
    cubic_coeff_a, exclude_outside, extrapolation_value = -0.75, 0, 0.0
    for attr in node.attribute:
        if attr.name == "mode": mode = attr.s.decode('utf-8')
        elif attr.name == "coordinate_transformation_mode": coord_mode = attr.s.decode('utf-8')
        elif attr.name == "nearest_mode": nearest_mode = attr.s.decode('utf-8')
        elif attr.name == "cubic_coeff_a": cubic_coeff_a = attr.f
        elif attr.name == "exclude_outside": exclude_outside = attr.i
        elif attr.name == "extrapolation_value": extrapolation_value = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Resize(
        node.input,
        node.output,
        mode=mode,
        coord_mode=coord_mode,
        nearest_mode=nearest_mode,
        cubic_coeff_a=cubic_coeff_a,
        exclude_outside=exclude_outside,
        extrapolation_value=extrapolation_value,
        dtype=onnx_dtype_mapping[elem_type],
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("GridSample")
def _factory_088_gridsample(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    mode, padding_mode, align_corners = "bilinear", "zeros", 0
    for attr in node.attribute:
        if attr.name == "mode": mode = attr.s.decode("utf-8")
        elif attr.name == "padding_mode": padding_mode = attr.s.decode("utf-8")
        elif attr.name == "align_corners": align_corners = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.GridSample(
        node.input,
        node.output,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        dtype=onnx_dtype_mapping[elem_type],
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("TopK")
def _factory_089_topk(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, largest, sorted_ = -1, 1, 1
    for attr in node.attribute:
        if attr.name == "axis": axis = attr.i
        elif attr.name == "largest": largest = attr.i
        elif attr.name == "sorted": sorted_ = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.TopK(node.input, node.output, axis=axis, largest=largest, sorted=sorted_, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("CumSum")
def _factory_090_cumsum(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    exclusive, reverse = 0, 0
    for attr in node.attribute:
        if attr.name == "exclusive": exclusive = attr.i
        elif attr.name == "reverse": reverse = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.CumSum(node.input, node.output, exclusive=exclusive, reverse=reverse, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("RandomUniformLike")
def _factory_091_randomuniformlike(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    high, low, seed, dtype_attr = 1.0, 0.0, 0.0, None
    for attr in node.attribute:
        if attr.name == "high": high = attr.f
        elif attr.name == "low": low = attr.f
        elif attr.name == "seed": seed = attr.f
        elif attr.name == "dtype": dtype_attr = attr.i
    target_dtype = onnx_dtype_mapping.get(dtype_attr, "float32") if dtype_attr is not None else None
    onnx_graph_list.append(nn.Operators.RandomUniformLike(node.input, node.output, high=high, low=low, seed=seed, dtype=target_dtype, version="17"))
    return onnx_graph_list[-1]


@register_factory("RandomUniform")
def _factory_092_randomuniform(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    high, low, seed, dtype_attr, shape = 1.0, 0.0, 0.0, 1, []
    for attr in node.attribute:
        if attr.name == "high": high = attr.f
        elif attr.name == "low": low = attr.f
        elif attr.name == "seed": seed = attr.f
        elif attr.name == "dtype": dtype_attr = attr.i
        elif attr.name == "shape": shape = attr.ints
    onnx_graph_list.append(nn.Operators.RandomUniform(node.input, node.output, high=high, low=low, seed=seed, dtype=dtype_attr, shape=shape, version="17"))
    return onnx_graph_list[-1]
