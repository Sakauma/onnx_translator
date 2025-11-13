import onnx
import numpy as np
from onnx import numpy_helper
import nn.Operators
from nn import onnx_dtype_mapping
from onnx import shape_inference


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


def ONNXImport(file_path):
    """
    从ONNX模型文件导入计算图节点
    
    Args:
        file_path: ONNX模型文件路径
        
    Returns:
        list: 包含操作节点的列表
    """
    onnx_graph_list = []
    
    # 加载ONNX模型
    onnx_model = onnx.load(file_path, load_external_data=False)
    
    # 提取初始化器的形状信息
    initializer_shapes = {}
    for init in onnx_model.graph.initializer:
        shape = [dim for dim in init.dims]
        initializer_shapes[init.name] = shape
        
    # 遍历图中的每个节点
    for node in onnx_model.graph.node:
        if node.op_type.upper() == "RELU":
            # 处理ReLU操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            # print("relu node data type", onnx_dtype_mapping[elem_type])
            onnx_graph_list.append(
                nn.Operators.__getattribute__("RELU")(node.input,
                                                      node.output,
                                                      dtype=onnx_dtype_mapping[elem_type],
                                                      version="17"))
        elif node.op_type.upper() == "COS":
            # 处理COS操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("COS")(node.input,
                                                     node.output,
                                                     dtype=onnx_dtype_mapping[elem_type],
                                                     version="17"))
        elif node.op_type.upper() == "ABS":
            # 处理ABS操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("ABS")(node.input,
                                                     node.output,
                                                     dtype=onnx_dtype_mapping[elem_type],
                                                     version="17"))  
        elif node.op_type.upper() == "ADD":
            # 处理ADD操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("ADD")(node.input,
                                                    node.output,
                                                    dtype=onnx_dtype_mapping[elem_type],
                                                    version="17"))
        elif node.op_type.upper() == "OTHER_OPS":
            # 其他操作节点的处理占位符
            pass
        else:
            # 忽略未支持的操作类型
            pass
            
    return onnx_graph_list
