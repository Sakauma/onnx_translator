# from nn import Graph
# import nn.ModelInitParas
# from nn.ONNXImport import ONNXImport
# from nn.GraphVisualization import GraphGenerate

# file_path = "./onnx_model/model_with_relu_and_cos_modified.onnx"
# model = "relu_cos"
# initial_graph = Graph(
#     ONNXImport(file_path),
#     nn.ModelInitParas.ONNXParasGen(file_path)[0],
#     model_name=model
# )

# """graph forward computing with real data"""
# print("inputs tensor data: ", nn.ModelInitParas.ONNXParasGen(file_path)[1][0].data)
# initial_graph.forward(*nn.ModelInitParas.ONNXParasGen(file_path)[1])
# GraphGenerate(initial_graph, model)

# """graph forward computing without real data"""
# print("inputs tensor data: ", nn.ModelInitParas.ONNXParasGen(file_path)[1][0].data)
# initial_graph.forward_(*nn.ModelInitParas.ONNXParasGen(file_path)[1])
# GraphGenerate(initial_graph, model)

from nn import Graph
import nn.ModelInitParas
from nn.ONNXImport import ONNXImport
from nn.GraphVisualization import GraphGenerate

# --- 准备工作 ---
file_path = "./onnx_model/model_with_relu_and_cos_modified.onnx"
model = "relu_cos"

# --- 只调用 ONNXParasGen 一次 ---
print("正在解析模型输入并生成一组张量...")
initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(file_path)
print("...完成。\n")


# --- 构建计算图 ---
print("正在构建计算图...")
initial_graph = Graph(
    ONNXImport(file_path),
    initial_inputs,  # 使用已保存的变量
    model_name=model
)
print("...计算图构建完毕。\n")


# --- 使用真实数据运行 ---
print("--- 正在使用真实数据运行 forward() ---")
# 只打印一小部分样本数据，避免刷屏
print("输入张量数据样本:", initial_tensors[0].data[0, 0, 0, :5])
initial_graph.forward(*initial_tensors) # 使用已保存的变量
GraphGenerate(initial_graph, model)
print("...带数据的正向计算完成。\n")


# --- 不使用真实数据运行 ---
# 注意：第一次 forward 调用后，图的内部状态可能已改变。
# 为了进行干净的测试，最好重新创建一个图。
print("--- 正在不使用真实数据运行 forward_() ---")
graph_for_sim = Graph(
    ONNXImport(file_path),
    initial_inputs,
    model_name=model
)
graph_for_sim.forward_(*initial_tensors) # 使用已保存的变量
# 我们可以重命名输出文件以避免覆盖
GraphGenerate(graph_for_sim, model + "_sim")
print("...不带数据的正向计算完成。\n")

print("✅ 测试脚本成功执行完毕！")

# import onnx
# import onnxruntime

# # 1. 检查 onnx 版本，确保是我们指定的 1.19
# print(f"ONNX library version: {onnx.__version__}")

# # 2. 检查 onnxruntime 版本
# print(f"ONNX Runtime version: {onnxruntime.__version__}")

# # 3. 检查可用的执行提供程序（Providers）
# available_providers = onnxruntime.get_available_providers()
# print(f"Available execution providers: {available_providers}")

# # 4. 最关键的一步：确认 'CUDAExecutionProvider' 是否在列表中
# if 'CUDAExecutionProvider' in available_providers:
#     print("\n✅ Success! ONNX Runtime for GPU is correctly installed and can see your CUDA device.")
# else:
#     print("\n⚠️ Warning! 'CUDAExecutionProvider' not found. The GPU version may not be working correctly.")
#     print("   Please check your CUDA and cuDNN installation and path.")