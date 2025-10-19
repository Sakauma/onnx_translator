import torch
import torch.nn as nn


class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()

    def forward(self, x:torch.Tensor,y:torch.Tensor) -> torch.Tensor:
        added = x + y
        return added

if __name__ == "__main__":
    # 1. 实例化模型并设置为评估模式
    model = AddModel()
    model.eval()

    # 2. 创建一个符合 ONNX 导出要求的虚拟输入张量
    #    形状可以自定义，这里使用一个常见的 (batch_size, channels, height, width) 格式
    dummy_input_a = torch.randn(1, 3, 32, 32)
    dummy_input_b = torch.randn(1, 3, 32, 32)
    # 3. 定义 ONNX 文件的名称
    onnx_file_name = "onnx_model/model_with_add.onnx"

    # 4. 执行导出操作
    print(f"正在导出模型到 {onnx_file_name}...")
    torch.onnx.export(
        model,                  # 要导出的模型
        (dummy_input_a, dummy_input_b),
        onnx_file_name,         # 输出文件名
        input_names=["input1", "input2"],  # ONNX图中输入节点的名称
        output_names=["output"],# ONNX图中输出节点的名称
        opset_version=17,       # ONNX算子集版本，根据你的项目要求设为17
        verbose=True            # 打印详细的导出信息
    )

    print(f"\n✅ 模型已成功导出为 '{onnx_file_name}'！")
