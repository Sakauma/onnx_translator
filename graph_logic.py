# 运行一个ONNX模型通过完整的导入器（ONNXImport -> ModelInitParas -> Graph），
# 并生成可视化的流程图，以验证图的结构、节点连接是否正确。

import nn
from nn import Graph
import nn.ModelInitParas
from nn.ONNXImport import ONNXImport
from nn.GraphVisualization import GraphGenerate
import os
import numpy as np

# --- 准备工作 ---
# 生成一个包含所有已实现算子的模型
file_path = "./onnx_model/model.onnx"
model_name = "graph_logic_test"

# 创建结果目录
result_dir = os.path.join("./result", model_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f"创建目录: {result_dir}")

# --- 核心流程 ---
# 1. 导入算子
print("步骤 1: 导入 ONNX 模型并映射算子...")
ops_list = ONNXImport(file_path)
if not ops_list:
    print("错误：没有从 ONNX 文件中导入任何算子。")
    exit(1)
print(f"成功导入 {len(ops_list)} 个算子。")

# 2. 解析输入参数
print("\n步骤 2: 解析模型初始输入参数...")
try:
    initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(file_path)
    print(f"模型输入: {initial_inputs}")
    print(f"生成模拟输入张量 (形状): {[t.size for t in initial_tensors]}")
except Exception as e:
    print(f"解析输入参数时出错: {e}")
    exit(1)
    
# 3. 实例化计算图
print("\n步骤 3: 创建 Graph 对象 (用于图结构推断)...")
# 我们使用 forward_ 来进行无数据的图推断
# 这需要一个单独的图实例
graph_for_logic = Graph(
    ops=ONNXImport(file_path), # 重新导入以获得干净的op实例
    input_name=initial_inputs,
    model_name=model_name
)

# 使用 Tensor_ (占位符) 进行推断
placeholder_tensors = [nn.Tensor_(*t.size, dtype=t.dtype) for t in initial_tensors]
graph_for_logic.forward_(*placeholder_tensors)
print("图结构推断 (forward_) 完成。")


# 4. 生成并保存计算图的可视化结果
print("\n步骤 4: 生成图可视化文件...")
try:
    GraphGenerate(graph_for_logic, model_name)
    print(f"✅ 逻辑验证成功！流程图已保存在 '{result_dir}' 目录中。")
except Exception as e:
    print(f"❌ 生成图表时出错: {e}")