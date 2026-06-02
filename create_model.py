"""文件功能：构造覆盖当前支持算子集合的 PyTorch 模型，并导出为 ONNX 示例模型。
作者：Egor Izmaylov
时间：2026-06-02
"""

import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
import os
import warnings

class FinalSupportedModel(nn.Module):
    """
    测试模型。
    """
    # 初始化 `FinalSupportedModel` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self):
        super(FinalSupportedModel, self).__init__()
        
        # --- 1. 卷积与归一化 ---
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
        
        # 归一化 
        self.bn = nn.BatchNorm2d(8)
        self.in_norm = nn.InstanceNorm2d(8, track_running_stats=True)
        self.ln = nn.LayerNorm([8, 32, 32]) 
        # self.gn = nn.GroupNorm(4, 8) 
        
        # --- 2. 激活函数 ---
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        self.softplus = nn.Softplus()
        self.gelu = nn.GELU()
        
        # --- 3. 池化 ---
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- 4. 全连接 ---
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 128) # 8*8*8=512
        
        # Buffer
        self.register_buffer('dummy_idx', torch.tensor([0], dtype=torch.int64))

    # 执行 `FinalSupportedModel` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        # x: [1, 4, 32, 32]
        results = []
        
        # === 1. 多分支特征提取 ===
        # 分支 A: Conv -> BN -> ReLU
        x1 = self.conv1(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        
        # 分支 B: Conv -> IN -> LeakyReLU
        x2 = self.conv1(x)
        x2 = self.in_norm(x2)
        x2 = self.leaky_relu(x2)
        
        # 分支 C: Conv -> LN -> Tanh
        x3 = self.conv1(x)
        x3 = self.ln(x3)
        x3 = self.tanh(x3)
        
        # [已移除 GN 分支，直接复用卷积结果做 Sigmoid]
        x4 = self.conv1(x)
        x4 = self.sigmoid(x4)
        
        # === 2. 融合 ===
        # 此时所有分支 shape 均为 [1, 8, 32, 32]
        out = x1 + x2 + x3 + x4
        
        # === 3. 后处理 ===
        out_max = self.maxpool(out) # 16x16
        out_avg = self.avgpool(out_max) # 8x8
        
        flat = self.flatten(out_avg) # [1, 512]
        fc_out = self.fc(flat)       # [1, 128]
        results.append(fc_out.sum().view(1))
        
        # === 4. 形状变换验证 ===
        reshaped = fc_out.view(1, 8, 16) 
        transposed = reshaped.transpose(1, 2)
        results.append(transposed.sum().view(1))
        
        # === 5. 数学运算 ===
        y = x * 0.5
        results.append(torch.add(x, y).sum().view(1))
        results.append(torch.sub(x, y).sum().view(1))
        results.append(torch.mul(x, y).sum().view(1))
        results.append(torch.div(x, y + 1.0).sum().view(1))
        
        results.append(torch.abs(x).sum().view(1))
        results.append(torch.pow(torch.abs(x), 2).sum().view(1))
        results.append(torch.sin(x).sum().view(1))
        results.append(torch.clamp(x, -0.5, 0.5).sum().view(1))

        # === 6. 其他操作 ===
        cat_res = torch.cat([x, x], dim=1) 
        split_res = torch.chunk(cat_res, 2, dim=1) 
        results.append(split_res[0].sum().view(1))
        
        expand_res = x[:, 0:1, :, :].expand(1, 4, 32, 32)
        results.append(expand_res.sum().view(1))
        
        results.append(self.global_avg(x).sum().view(1))
        
        # 汇总
        final_out = torch.cat(results, dim=0).sum()
        return final_out

# 实现 `export_model` 步骤，规范化输入并返回下游期望的数据或元信息。
def export_model():
    model = FinalSupportedModel()
    model.eval()
    dummy_input = torch.randn(1, 4, 32, 32)
    output_dir = "./onnx_model"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "model.onnx")
    
    print(f"🚀 正在导出最终模型 (移除 GroupNorm) 到: {output_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            model, dummy_input, output_path,
            opset_version=17,
            input_names=['input'], output_names=['output'],
            do_constant_folding=False,
            dynamo=False,
        )
    print("✅ 导出成功！")

if __name__ == "__main__":
    export_model()
