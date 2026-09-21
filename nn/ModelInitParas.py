# /**
#   ******************************************************************************
#   * @file        ModelInitParas.py
#   * @author      Egor Izmaylov
#   * @brief       提取 ONNX 模型 initializer 的 dtype 与参数信息，辅助模型导入和检查。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import onnx
from nn import Tensor
from nn import onnx_dtype_mapping
import numpy as np
from onnx import numpy_helper, shape_inference
from nn.importer.model_loader import load_model

# ONNX数据类型到NumPy数据类型的映射
onnx_np_dtype_mapping = {
 "float32": np.float32,
 "float16": np.float16,
 "int64": np.int64,
 "int32": np.int32,
 "bool": np.bool_,
}

# 实现 `get_tensor_dtype` 步骤，规范化输入并返回下游期望的数据或元信息。
def get_tensor_dtype(tensor_name, model):
    """
    获取张量的数据类型
    """
    # 对模型进行形状推断
    try:
        inferred_model = shape_inference.infer_shapes(model)
        graph = inferred_model.graph
    except:
        graph = model.graph
    
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

# 实现 `ONNXParasGen` 步骤，规范化输入并返回下游期望的数据或元信息。
def ONNXParasGen(file_path):
    """
    从ONNX模型文件生成初始参数张量 (已修复权重冲突问题)
    
    Args:
        file_path: ONNX模型文件路径
        
    Returns:
        tuple: (输入列表, 张量列表)
    """
    inputs_list = []
    tensor_list = []
    
    # 加载ONNX模型
    model = load_model(file_path, load_external_data=True)
    graph = model.graph

    # 只在 initializer 未声明为 graph input 时将其视为纯内部常量。
    # 同名 graph input 的 initializer 是 ONNX 定义的可覆盖默认值。
    initializers = {init.name: init for init in graph.initializer}
    
    # 遍历图的输入节点
    for item in graph.input:
        print("item: ", item.name)
        inputs_list.append(item.name)
        
        # 提取张量维度信息
        dimensions = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        dimensions = [d if (d is not None and d > 0) else 1 for d in dimensions]

        # print("initial tensor dtype:", get_tensor_dtype(item.name, model))
        # 获取张量数据类型
        elem_type = get_tensor_dtype(item.name, model)
        dtype = onnx_dtype_mapping.get(elem_type, "float32")

        if item.name in initializers:
            # 保持 names/tensors 等长，供现有 CLI 按位置 zip；传给 Graph 的
            # 数值与省略该参数时使用的 serialized default 完全相同。
            data = np.asarray(numpy_helper.to_array(initializers[item.name])).copy()
            tensor_list.append(Tensor(*data.shape, dtype=dtype, data=data))
            continue
        
        # 根据数据类型创建随机张量
        if "float" in dtype:
            tensor = Tensor(*dimensions, dtype=dtype)
            # 确保使用对应的 numpy 类型生成数据
            np_dtype = onnx_np_dtype_mapping.get(dtype, np.float32)
            tensor.data = np.random.rand(*dimensions).astype(np_dtype)
            tensor_list.append(tensor)
        else:
            tensor = Tensor(*dimensions, dtype=dtype)
            np_dtype = onnx_np_dtype_mapping.get(dtype, np.int32)
            tensor.data = np.random.randint(0, 2, size=dimensions).astype(np_dtype)
            tensor_list.append(tensor)
            
    return inputs_list, tensor_list
