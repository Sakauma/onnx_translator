import nn
from nn import Graph
import nn.ModelInitParas
from nn.ONNXImport import ONNXImport
from nn.GraphVisualization import GraphGenerate
import os

# --- 准备工作 ---
# ONNX模型文件路径
file_path = "./model.onnx"
# 模型名称，用于创建结果文件夹
model_name = "relu_cos_test"

# 创建结果目录（如果不存在）
result_dir = os.path.join("./result", model_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f"Created directory: {result_dir}")


# --- 核心流程 ---
# 1. 从ONNX文件导入算子，构建NPS算子列表
print("Step 1: Importing ONNX model and mapping operators...")
ops_list = ONNXImport(file_path)
print(f"Successfully imported {len(ops_list)} operators.")

# 2. 从ONNX文件解析模型的初始输入参数
print("\nStep 2: Generating initial parameters and input tensors...")
initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(file_path)
print("Initial input names:", initial_inputs)
print("Generated input tensor shape:", initial_tensors[0].size)

# 3. 使用导入的算子和输入来实例化计算图
print("\nStep 3: Creating the Graph object...")
initial_graph = Graph(
    ops=ops_list,
    input_name=initial_inputs,
    model_name=model_name
)
print("Graph created successfully.")

# 4. 执行图的前向计算（使用真实数据）
print("\nStep 4: Running graph forward pass with real data...")
# 使用 '*' 将 initial_tensors 列表解包为独立的参数传递给 forward
initial_graph.forward(*initial_tensors)
print("Forward pass completed.")

# 5. 生成并保存计算图的可视化结果
print("\nStep 5: Generating graph visualization...")
GraphGenerate(initial_graph, model_name)
print(f"Graph visualization saved in '{result_dir}' directory.")

# --- 验证（可选） ---
# 我们可以打印最终输出张量的一些信息来做初步验证
# 注意：Graph的forward实现目前没有返回值，但我们可以通过修改Graph类或访问其内部状态来获取输出
# 这是一个高级步骤，当前我们可以通过程序不报错并成功生成图片来初步判断成功

print("\n✅ Project pipeline executed successfully!")