import onnx
from nn import Tensor
from nn import onnx_dtype_mapping
import numpy as np
from onnx import shape_inference

# ONNX数据类型到NumPy数据类型的映射
onnx_np_dtype_mapping = {
 "float32": np.float32
}

def get_tensor_dtype(tensor_name, model):
    """
    获取张量的数据类型
    
    Args:
        tensor_name: 张量名称
        model: ONNX模型对象
        
    Returns:
        int: ONNX数据类型编码，如果未找到返回None
    """
    # 对模型进行形状推断
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph
    
    # 在图的输入中查找
    for input_tensor in graph.input:
        if input_tensor.name == tensor_name:
            return input_tensor.type.tensor_type.elem_type
            
    # 在图的输出中查找
    for output_tensor in graph.output:
        if output_tensor.name == tensor_name:
            return output_tensor.type.tensor_type.elem_type
            
    # 在图的value_info中查找
    for value_info_tensor in graph.value_info:
        if value_info_tensor.name == tensor_name:
            return value_info_tensor.type.tensor_type.elem_type
            
    return None

def ONNXParasGen(file_path):
    """
    从ONNX模型文件生成初始参数张量
    
    Args:
        file_path: ONNX模型文件路径
        
    Returns:
        tuple: (输入列表, 张量列表)
    """
    inputs_list = []
    tensor_list = []
    
    # 加载ONNX模型
    model = onnx.load(file_path, load_external_data=False)
    graph = model.graph
    
    # 遍历图的输入节点
    for item in graph.input:
        print("item: ", item.name)
        inputs_list.append(item.name)
        # 提取张量维度信息
        dimensions = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        print("initial tensor dtype:",get_tensor_dtype(item.name, model))
        # 获取张量数据类型
        elem_type = get_tensor_dtype(item.name, model)
        dtype = onnx_dtype_mapping[elem_type]
        
        # 根据数据类型创建随机张量
        if "float" in dtype:
            tensor = Tensor(*dimensions, dtype=dtype)
            tensor.data = np.random.rand(*dimensions).astype(onnx_np_dtype_mapping[dtype])
            tensor_list.append(tensor)
        else:
            tensor = Tensor(*dimensions, dtype=dtype)
            tensor.data = np.random.randint(-10, 10, size=dimensions, dtype=onnx_np_dtype_mapping[dtype])
            tensor_list.append(tensor)
            
    return inputs_list, tensor_list