from nn import Graph
import nn.ModelInitParas
from nn.ONNXImport import ONNXImport
from nn.GraphVisualization import GraphGenerate

# --- 准备工作 ---
file_path = "./onnx_model/model_with_reshape.onnx"
model = "reshape"

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
