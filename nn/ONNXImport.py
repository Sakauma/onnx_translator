import onnx
import numpy as np
from onnx import numpy_helper
import nn.Operators
from nn import onnx_dtype_mapping
from onnx import shape_inference


def get_tensor_dtype(tensor_name, model):
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph
    # Check in the graph's inputs
    for input_tensor in graph.input:
        if input_tensor.name == tensor_name:
            return input_tensor.type.tensor_type.elem_type
    # Check in the graph's outputs
    for output_tensor in graph.output:
        if output_tensor.name == tensor_name:
            return output_tensor.type.tensor_type.elem_type
# Check in the graph's value_info
    for value_info_tensor in graph.value_info:
        if value_info_tensor.name == tensor_name:
            return value_info_tensor.type.tensor_type.elem_type
    return None


def ONNXImport(file_path):
    onnx_graph_list = []
    onnx_model = onnx.load(file_path, load_external_data=False)
    initializer_shapes = {}
    for init in onnx_model.graph.initializer:
        shape = [dim for dim in init.dims]
        initializer_shapes[init.name] = shape
    for node in onnx_model.graph.node:
        if node.op_type.upper() == "RELU":
            elem_type=get_tensor_dtype(node.output[0], onnx_model)
# print("relu node data type", onnx_dtype_mapping[elem_type])
            onnx_graph_list.append(
                nn.Operators.__getattribute__("RELU")(node.input,
                                                      node.output,
                                                      dtype=onnx_dtype_mapping[elem_type],
                                                      version="17"))
        elif node.op_type.upper() == "COS":
            elem_type=get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("COS")(node.input,
                                                     node.output,
                                                     dtype=onnx_dtype_mapping[elem_type],
                                                     version="17"))
        elif node.op_type.upper() == "OTHER_OPS":
            pass
        else:
            pass
    return onnx_graph_list
