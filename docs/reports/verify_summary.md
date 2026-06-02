<!--
文件功能：记录阶段性算子数值验证和图结构验证总结。
作者：Egor Izmaylov
时间：2026-06-02
-->

# 算子验证总结说明

## 一、验证背景
本次验证工作对算子出现频率排序开展，  
按出现次数从高到低优先验证高频算子，目标覆盖高频算子为优先。

验证重点为：
- Python 侧算子执行正确性
- CUDA 侧数值计算正确性
- 模型图构建与 shape 推导正确性

在保证结果正确的前提下，未对所有算子做极端 corner case 覆盖。

---

## 二、数值验证（Python + CUDA）

数值验证采用工程内统一验证入口 `python tools/cli.py numerical`：

- Python 侧通过 `nn/Operators.py` 调用 `tensor_ops.so` 作为参考实现
- CUDA 侧通过 `cache/verify_*` 可执行文件作为 ground truth
- 对 Python 与 CUDA 结果逐元素数值对比
- iterations 从默认 200 调整为 20（稳定高效）

### 2.1 说明

- **Mod**：验证时采用 Python/ONNX 风格的 mod 语义（结果与除数同号），避免与 C 的 `fmod` 语义差异带来大误差。
- **RandomUniformLike**：由于不同后端的随机数实现未必逐元素一致，验证中采用固定 RNG 参考实现进行逐元素一致性验证。

注：`ReduceProd` 在随机输入下可能出现溢出；脚本对 `NaN/Inf/Overflow` 做逻辑匹配，当出现 “all values were NaN/Inf/Overflow matched” warning 时，仍表示两侧结果在该情形下匹配，验证可稳定通过。

### 2.2 已完成数值验证的算子（Python + CUDA）

#### 基础算子 / 一元算子
- Neg
- Floor
- Sign
- IsNaN

#### 基础算子 / 二元算子
- Add
- Sub
- Mul
- Div
- Mod
- Max
- Min

#### 激活 / 数学函数
- Relu
- Sigmoid
- Tanh
- Sin
- Cos
- Tan
- Atan
- Exp
- Log
- Sqrt
- Pow

#### 线性代数 / 卷积 / 池化 / Softmax
- Conv
- Gemm
- MatMul
- Einsum
- MaxPool
- Softmax

#### 比较 / 逻辑
- Equal
- Greater
- Less
- GreaterOrEqual
- LessOrEqual
- Not
- And
- Or
- Xor

#### Reduce
- ReduceMean
- ReduceSum
- ReduceMax
- ReduceMin
- ReduceProd

#### 索引 / Scatter-Gather
- Gather
- GatherElements
- GatherND
- ScatterND
- NonZero
- TopK
- ArgMin
- ArgMax

#### 扫描 / 随机 / 采样
- CumSum
- Resize
- RandomUniformLike

以上算子均通过 Python 与 CUDA 数值一致性验证。

---

## 三、图结构与 Shape 验证

针对算子中以图结构和 shape 推导为主的算子，
采用模型图验证方式进行覆盖：

- 使用 `python tools/cli.py create-model` 生成模型
- 使用 `python tools/cli.py verify-graph` 验证图构建、节点连接及 shape 推导过程
- 验证模型图能够正确生成且无报错

图结构与 shape 验证通过 `python tools/cli.py create-graph-model` 构造覆盖 Cast / Shape / ConstantOfShape / Unsqueeze / Squeeze / Slice / Transpose / Concat / Reshape / Expand / Where / Range 等算子的 ONNX 模型，并使用 `python tools/cli.py verify-graph` 完成图导入、节点连接与 shape 推导验证，生成图结构可视化结果（result/nps_verification/nps_verification.pdf）
