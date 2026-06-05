# /**
#   ******************************************************************************
#   * @file        node_factories_04.py
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


@register_factory("TfIdfVectorizer")
def _factory_141_tfidfvectorizer(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    attrs = {
        "mode": None,
        "ngram_counts": None,
        "ngram_indexes": None,
        "max_skip_count": None,
        "min_gram_length": None,
        "max_gram_length": None,
        "pool_int64s": [],
        "pool_strings": [],
        "weights": [],
    }
    for attr in node.attribute:
        if attr.name == "mode": attrs["mode"] = attr.s.decode("utf-8")
        elif attr.name == "ngram_counts": attrs["ngram_counts"] = list(attr.ints)
        elif attr.name == "ngram_indexes": attrs["ngram_indexes"] = list(attr.ints)
        elif attr.name == "max_skip_count": attrs["max_skip_count"] = attr.i
        elif attr.name == "min_gram_length": attrs["min_gram_length"] = attr.i
        elif attr.name == "max_gram_length": attrs["max_gram_length"] = attr.i
        elif attr.name == "pool_int64s": attrs["pool_int64s"] = list(attr.ints)
        elif attr.name == "pool_strings": attrs["pool_strings"] = [item.decode("utf-8") for item in attr.strings]
        elif attr.name == "weights": attrs["weights"] = list(attr.floats)
    required = ["mode", "ngram_counts", "ngram_indexes", "max_skip_count", "min_gram_length", "max_gram_length"]
    missing = [name for name in required if attrs[name] is None]
    if missing:
        raise ValueError(f"TfIdfVectorizer missing required attribute(s): {missing}")
    onnx_graph_list.append(nn.Operators.TfIdfVectorizer(
        node.input,
        node.output,
        mode=attrs["mode"],
        ngram_counts=attrs["ngram_counts"],
        ngram_indexes=attrs["ngram_indexes"],
        max_skip_count=attrs["max_skip_count"],
        min_gram_length=attrs["min_gram_length"],
        max_gram_length=attrs["max_gram_length"],
        pool_int64s=attrs["pool_int64s"],
        pool_strings=attrs["pool_strings"],
        weights=attrs["weights"],
        version="17",
    ))
    return onnx_graph_list[-1]


@register_factory("Binarizer")
def _factory_142_binarizer(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    threshold = 0.0
    for attr in node.attribute:
        if attr.name == "threshold": threshold = attr.f
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.Binarizer(node.input, node.output, threshold=threshold, dtype=onnx_dtype_mapping[elem_type], version="17"))
    return onnx_graph_list[-1]


@register_factory("DynamicQuantizeLinear")
def _factory_143_dynamicquantizelinear(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    onnx_graph_list.append(nn.Operators.DynamicQuantizeLinear(node.input, node.output, version="17"))
    return onnx_graph_list[-1]


@register_factory("AffineGrid")
def _factory_144_affinegrid(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    align_corners = 0
    for attr in node.attribute:
        if attr.name == "align_corners": align_corners = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(nn.Operators.AffineGrid(node.input, node.output, align_corners=align_corners, dtype=onnx_dtype_mapping[elem_type], version="20"))
    return onnx_graph_list[-1]


@register_factory("TensorScatter")
def _factory_145_tensorscatter(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    axis, mode = -2, "linear"
    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i
        elif attr.name == "mode":
            mode = attr.s.decode("utf-8")
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(
        nn.Operators.TensorScatter(
            node.input,
            node.output,
            axis=axis,
            mode=mode,
            dtype=onnx_dtype_mapping[elem_type],
            version="24",
        )
    )
    return onnx_graph_list[-1]


@register_factory("RegexFullMatch")
def _factory_146_regexfullmatch(node, import_context):
    onnx_graph_list = []
    pattern = None
    for attr in node.attribute:
        if attr.name == "pattern":
            pattern = attr.s.decode("utf-8")
    onnx_graph_list.append(nn.Operators.RegexFullMatch(node.input, node.output, pattern=pattern, version="20"))
    return onnx_graph_list[-1]


@register_factory("StringConcat")
def _factory_147_stringconcat(node, import_context):
    onnx_graph_list = []
    onnx_graph_list.append(nn.Operators.StringConcat(node.input, node.output, version="20"))
    return onnx_graph_list[-1]


@register_factory("StringSplit")
def _factory_148_stringsplit(node, import_context):
    onnx_graph_list = []
    delimiter, maxsplit = None, None
    for attr in node.attribute:
        if attr.name == "delimiter":
            delimiter = attr.s.decode("utf-8")
        elif attr.name == "maxsplit":
            maxsplit = attr.i
    onnx_graph_list.append(
        nn.Operators.StringSplit(
            node.input,
            node.output,
            delimiter=delimiter,
            maxsplit=maxsplit,
            version="20",
        )
    )
    return onnx_graph_list[-1]


@register_factory("ImageDecoder")
def _factory_149_imagedecoder(node, import_context):
    onnx_graph_list = []
    pixel_format = "RGB"
    for attr in node.attribute:
        if attr.name == "pixel_format":
            pixel_format = attr.s.decode("utf-8")
    onnx_graph_list.append(nn.Operators.ImageDecoder(node.input, node.output, pixel_format=pixel_format, version="20"))
    return onnx_graph_list[-1]


@register_factory("RotaryEmbedding")
def _factory_150_rotaryembedding(node, import_context):
    get_dtype = lambda name, default=onnx.TensorProto.FLOAT: import_context.get_dtype(name, default)
    onnx_graph_list = []
    num_heads, rotary_embedding_dim, interleaved = 0, 0, 0
    for attr in node.attribute:
        if attr.name == "num_heads":
            num_heads = attr.i
        elif attr.name == "rotary_embedding_dim":
            rotary_embedding_dim = attr.i
        elif attr.name == "interleaved":
            interleaved = attr.i
    elem_type = get_dtype(node.output[0])
    onnx_graph_list.append(
        nn.Operators.RotaryEmbedding(
            node.input,
            node.output,
            num_heads=num_heads,
            rotary_embedding_dim=rotary_embedding_dim,
            interleaved=interleaved,
            dtype=onnx_dtype_mapping[elem_type],
            version="23",
        )
    )
    return onnx_graph_list[-1]
