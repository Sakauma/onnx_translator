### 1. 测试环境

在开始之前，请确保你已安装以下工具：
* `gcc` (用于编译 C 共享库)
* `nvcc` (NVIDIA CUDA 工具包，用于编译 CUDA 验证程序)
* `python`
* Python 依赖库: `numpy`, `torch`, `onnx`

### 2. 步骤一：编译 C 后端共享库

此步骤将 `tensor_ops/tensor_ops.c` 中实现的C算子编译为一个共享库

在项目根目录运行：
```bash
# -fPIC: 生成位置无关代码
# -shared: 创建共享库
# -lm: 链接数学库
gcc -shared -o tensor_ops.so -fPIC tensor_ops/tensor_ops.c -lm
```

### 3. 步骤二：编译 CUDA 验证程序

在项目根目录运行：
```bash
nvcc cuda/verify_relu.cu -o verify_relu
nvcc cuda/verify_abs.cu -o verify_abs
nvcc cuda/verify_add.cu -o verify_add
nvcc cuda/verify_cos.cu -o verify_cos
```

### 4. 步骤三：生成 ONNX 测试模型

此步骤运行 `create_model.py` 脚本，使用 PyTorch 导出一个包含所有算子的测试模型。

```bash
python create_model.py
```

### 5. 步骤四：运行图逻辑验证

此脚本测试 `ONNXImport.py` 是否能正确解析 `model.onnx` 文件，并将所有算子正确映射和连接。

```bash
python graph_logic.py
```
### 6. 步骤五：运行数值正确性验证

此脚本测试验证C库中每个算子的功能、广播、类型提升和计算精度。

```bash
python numerical_correctness.py
```