# from nn import Graph
# import nn.ModelInitParas
# from nn.ONNXImport import ONNXImport
# from nn.GraphVisualization import GraphGenerate
#
# file_path = "./onnx_model/model_with_relu_and_cos_modified.onnx"
# model = "relu_cos"
# initial_graph = Graph(
#     ONNXImport(file_path),
#     nn.ModelInitParas.ONNXParasGen(file_path)[0],
#     model_name=model
# )
#
# """graph forward computing with real data"""
# print("inputs tensor data: ", nn.ModelInitParas.ONNXParasGen(file_path)[1][0].data)
# initial_graph.forward(*nn.ModelInitParas.ONNXParasGen(file_path)[1])
# GraphGenerate(initial_graph, model)
#
# """graph forward computing without real data"""
# print("inputs tensor data: ", nn.ModelInitParas.ONNXParasGen(file_path)[1][0].data)
# initial_graph.forward_(*nn.ModelInitParas.ONNXParasGen(file_path)[1])
# GraphGenerate(initial_graph, model)

import onnx
import onnxruntime

# 1. 检查 onnx 版本，确保是我们指定的 1.19
print(f"ONNX library version: {onnx.__version__}")

# 2. 检查 onnxruntime 版本
print(f"ONNX Runtime version: {onnxruntime.__version__}")

# 3. 检查可用的执行提供程序（Providers）
available_providers = onnxruntime.get_available_providers()
print(f"Available execution providers: {available_providers}")

# 4. 最关键的一步：确认 'CUDAExecutionProvider' 是否在列表中
if 'CUDAExecutionProvider' in available_providers:
    print("\n✅ Success! ONNX Runtime for GPU is correctly installed and can see your CUDA device.")
else:
    print("\n⚠️ Warning! 'CUDAExecutionProvider' not found. The GPU version may not be working correctly.")
    print("   Please check your CUDA and cuDNN installation and path.")