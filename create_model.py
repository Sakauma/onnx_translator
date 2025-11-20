# create_model.py
import torch
import torch.nn as nn
import os

class ComprehensiveModel(nn.Module):
    """
    一个包含 Relu, Abs, Add, Cos 的测试模型。
    它接收两个输入。
    """
    def __init__(self):
        super(ComprehensiveModel, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播路径:
        z = Cos( Relu(x) + Abs(y) )
        """
        # 1. 应用 ReLU
        x_relu = self.relu(x)
        
        # 2. 应用 Abs
        y_abs = torch.abs(y)
        
        # 3. 应用 Add (会产生广播)
        # 假设 x 的形状是 (1, 3, 32, 32)
        # 假设 y 的形状是 (1, 1, 1, 32)
        added = x_relu + y_abs
        
        # 4. 应用 Cos
        cos_output = torch.cos(added)
	
	# 5. 应用 Tanh
       	tanh_output = torch.tanh(cos_output)

        # 6. 应用 Reshape: 从 (1, 3, 32, 32) 重塑为 (1, 3, 1024)
        reshaped = torch.reshape(tanh_output, (1, 3, 1024))

        # 7. 应用 Unsqueeze: 在维度1上增加一个维度 (1, 3, 1024) -> (1, 1, 3, 1024)
        output = torch.unsqueeze(reshaped, 1)

        return output

# --- 主执行流程 ---
if __name__ == "__main__":
    # 1. 实例化模型并设置为评估模式
    model = ComprehensiveModel()
    model.eval()

    # 2. 创建两个符合广播规则的虚拟输入张量
    dummy_input_x = torch.randn(1, 3, 32, 32)
    dummy_input_y = torch.randn(1, 1, 1, 32) # 这个张量将会被广播

    # 3. 定义 ONNX 文件的名称
    onnx_dir = "onnx_model"
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_file_name = os.path.join(onnx_dir, "model.onnx")

    # 4. 执行导出操作
    print(f"正在导出模型到 {onnx_file_name}...")
    torch.onnx.export(
        model,
        (dummy_input_x, dummy_input_y), # 传入元组作为多个输入
        onnx_file_name,
        input_names=["input_x", "input_y"],  # 两个输入节点名称
        output_names=["output"],             # 输出节点名称
        opset_version=17,
        verbose=True
    )

    print(f"\n✅ 模型已成功导出为 '{onnx_file_name}'！")
