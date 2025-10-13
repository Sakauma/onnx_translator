import onnx
from nn import Tensor
from nn import onnx_dtype_mapping
import numpy as np
from onnx import shape_inference

onnx_np_dtype_mapping = {
 "float32": np.float32
}


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


def ONNXParasGen(file_path):
    inputs_list = []
    tensor_list = []
    model = onnx.load(file_path, load_external_data=False)
    graph = model.graph
    for item in graph.input:
        print("item: ", item.name)
    inputs_list.append(item.name)
    dimensions = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
    print("initial tensor dtype:",get_tensor_dtype(item.name, model))
    elem_type = get_tensor_dtype(item.name, model)
    dtype = onnx_dtype_mapping[elem_type]
    if "float" in dtype:
        tensor = Tensor(*dimensions, dtype=dtype)
        tensor.data = np.random.rand(*dimensions).astype(onnx_np_dtype_mapping[dtype])
        tensor_list.append(tensor)
    else:
        tensor = Tensor(*dimensions, dtype=dtype)
        tensor.data = np.random.randint(-10, 10, size=(dimensions), dtype=onnx_np_dtype_mapping[dtype])
        tensor_list.append(tensor)
    return inputs_list, tensor_list
