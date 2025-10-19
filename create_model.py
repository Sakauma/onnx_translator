# import torch
# import torch.nn as nn

# # 定义一个简单的模型，包含 ReLU 和 Cos
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         # ReLU 激活函数
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # x -> ReLU -> y
#         y = self.relu(x)
#         # y -> Cos -> z
#         z = torch.cos(y)
#         return z

# # 实例化模型
# model = SimpleModel()
# model.eval()

# # 创建一个虚拟输入张量
# dummy_input = torch.randn(1, 3, 16, 16) # 形状为 (1, 3, 16, 16)

# # 定义输入输出名称
# input_names = ["input_tensor"]
# output_names = ["output_tensor"]

# # 导出到 ONNX
# torch.onnx.export(model,
#                   dummy_input,
#                   "model.onnx",
#                   verbose=True,
#                   input_names=input_names,
#                   output_names=output_names,
#                   opset_version=17) # opset版本必须是17

# print("model.onnx has been created successfully!")

import torch
import torch.nn as nn

class ReluCosModel(nn.Module):
    """
    一个简单的模型，依次执行 ReLU 和 Cos 操作。
    """
    def __init__(self):
        super(ReluCosModel, self).__init__()
        # ReLU 激活函数层
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播路径。
        """
        # 1. 应用 ReLU 激活函数
        x_relu = self.relu(x)
        # 2. 应用 Cosine 函数
        x_cos = torch.cos(x_relu)
        return x_cos

# --- 主执行流程 ---
if __name__ == "__main__":
    # 1. 实例化模型并设置为评估模式
    model = ReluCosModel()
    model.eval()

    # 2. 创建一个符合 ONNX 导出要求的虚拟输入张量
    #    形状可以自定义，这里使用一个常见的 (batch_size, channels, height, width) 格式
    dummy_input = torch.randn(1, 3, 32, 32)

    # 3. 定义 ONNX 文件的名称
    onnx_file_name = "model_with_relu_and_cos_modified.onnx"

    # 4. 执行导出操作
    print(f"正在导出模型到 {onnx_file_name}...")
    torch.onnx.export(
        model,                  # 要导出的模型
        dummy_input,            # 模型的虚拟输入
        onnx_file_name,         # 输出文件名
        input_names=["input"],  # ONNX图中输入节点的名称
        output_names=["output"],# ONNX图中输出节点的名称
        opset_version=17,       # ONNX算子集版本，根据你的项目要求设为17
        verbose=True            # 打印详细的导出信息
    )

    print(f"\n✅ 模型已成功导出为 '{onnx_file_name}'！")