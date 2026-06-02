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


