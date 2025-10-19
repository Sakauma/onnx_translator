import torch
import torch.nn as nn


class ReshapeModel(nn.Module):
    def __init__(self):
        super(ReshapeModel, self).__init__()

    def forward(self, x: torch.Tensor, shape: torch.Tensor)->torch.Tensor:
        shape = list(shape)
        return torch.reshape(x, shape)


if __name__ == "__main__":
    # 1. 实例化模型
    model = ReshapeModel()
    model.eval()

    # 2. 创建输入
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_shape = torch.tensor([1, 3, 2, -1], dtype=torch.int64)

    # 3. 导出模型 - 使用传统导出方法
    onnx_file_name = "onnx_model/model_with_reshape.onnx"

    print(f"正在导出动态Reshape模型到 {onnx_file_name}...")

    # 方法1：禁用 dynamo
    torch.onnx.export(
        model,
        (dummy_input, dummy_shape),
        onnx_file_name,
        input_names=["input", "shape"],
        output_names=["output"],
        opset_version=18,
        verbose=True
    )

    print(f"\n✅ 动态Reshape模型已成功导出为 '{onnx_file_name}'！")

    # 验证输出
    with torch.no_grad():
        output = model(dummy_input, dummy_shape)
        print(f"输入数据形状: {dummy_input.shape}")
        print(f"目标形状: {tuple(dummy_shape.tolist())}")
        print(f"输出形状: {output.shape}")