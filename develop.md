# 算子开发标准流程清单

开发一个新算子需要修改以下 **5 个位置** 的代码：

---

## 1. C 后端接口 (`tensor_ops/tensor_ops.h`)
这是 Python 与 C 交互的接口。

* **定义参数结构体**：如果算子参数超过 3-4 个（如 Conv/Pool），建议定义一个 `struct` 来打包参数。
* **声明前向函数**：`void xxxx_forward(...)`。

## 2. C 后端实现 (`tensor_ops/tensor_ops.c`)
这是核心计算逻辑，**精度优先**。

* **复用辅助函数**：使用 `get_value_as_double` 读取输入，使用 `set_tensor_value_from_float` 写入输出。
* **核心逻辑**：
    * 解析输入形状和参数。
    * 使用多重循环遍历输出 Tensor 的每个元素。
    * **计算逻辑**：使用 `double` 进行高精度中间计算。
    * **并行化**：在最外层循环加上 `#pragma omp parallel for`。
* **边界处理**：注意处理 Padding、Stride 以及可能的越界情况。

## 3. Python 算子封装 (`nn/Operators.py`)
这是 NPS 前端定义。

* **定义 ctypes 结构**：如果 C 端定义了 struct，这里要定义对应的 `ctypes.Structure`。
* **新建算子类**：继承自 `Ops`。
    * `__init__`：保存参数，并设置 `self.lib.new_op_forward.argtypes`（**这一步很容易忘！**）。
    * `forward` (真实计算)：
        1.  推断输出形状 `out_shape`。
        2.  使用 `_numpy_to_ctensor` 转换输入。
        3.  打包参数（如果是 struct）。
        4.  调用 `self.lib.new_op_forward`。
        5.  使用 `_ctensor_to_numpy` 转回结果并释放 C Tensor 内存。
    * `forward_` (形状推断)：仅复制 `forward` 中的 `out_shape` 计算逻辑，返回占位符 `Tensor_`。

## 4. ONNX 解析适配 (`nn/ONNXImport.py`)
这是模型导入逻辑。

* 在循环中添加 `elif node.op_type == "NewOp":` 分支。
* **提取属性**：遍历 `node.attribute`，提取 `pads`, `strides`, `axis` 等参数。
* **实例化**：调用 `nn.Operators.NewOp(...)` 将节点加入图列表。

## 5. 验证 (`cuda/` & `numerical_correctness.py`)
确保结果正确。

* **编写 CUDA 真值 (`cuda/verify_new_op.cu`)**：
    * 使用 `double` 编写 Kernel，逻辑要求标准（不追求速度，只求对）。
    * 在 `main` 函数中处理参数读取（如果是复杂算子，需读取 `params.bin`）。
    * **记住一定使用 `malloc` 和 `free`！**
* **更新验证脚本 (`numerical_correctness.py`)**：
    1.  `import` 新算子类。
    2.  在 `verify_op` 中添加该算子的 **参数打包逻辑**（将 Python 参数转为 bytes 传给 CUDA）。
    3.  在 `plans` 列表中添加测试用例（覆盖 float32, float16, float8 等混合精度场景）。

---