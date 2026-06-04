<!--
/**
  ******************************************************************************
  * @file        verify_summary.md
  * @author      Egor Izmaylov
  * @brief       记录阶段性算子数值验证和图结构验证总结。
  * @details     2026.06.02  V1.0.0  创建
  * @details     2026.06.05  V1.0.1  补充完整 numerical 收尾验证记录
  * @details     2026.06.05  V1.0.2  补充 Transpose CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.3  补充 Flatten 与 Reshape CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.4  补充 Tile 与 Concat CUDA 数值门禁记录
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
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

---

## 四、2026.06.05 完整 numerical 收尾记录

本轮收尾前已完成一次完整数值门禁：

- 命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`
- 结果：全部通过，无失败算子。
- 覆盖：默认 active numerical plan 覆盖 80 个唯一算子名称、224 条默认计划，其中混合精度计划 142 条。
- 覆盖 dtype：包含 float32、float16、bfloat16、float8_e4m3、float8_e5m2，以及 int8、uint8、int64、bool 等量化、索引和比较相关类型。
- 高风险算子：MaxRoiPool、RoiAlign、RNN、GRU、LSTM、DFT、STFT 已进入完整 numerical 默认门禁，并在本轮一轮验证中通过。

本轮同时修正 `QuantizeLinear` 在 float32 scale 下的 nearest-even 舍入细节：当输入和 scale 均不是 float64 时，C 后端使用 float32 中间值与 `rintf`，避免将 float32 scale 提升到 double 后改变 ONNX 参考实现的半点舍入结果。新增回归测试覆盖 `axis=-1` 的 per-axis uint8/int8 量化，并使用 ONNX `ReferenceEvaluator` 对齐官方参考输出。

后续补充 `Transpose` 的 CUDA 参考验证程序，并将 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划接入默认 numerical 门禁。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op transpose --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Flatten` 与 `Reshape` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`Reshape` 的数值计划使用 `[0, -1]` 目标 shape，覆盖 ONNX 中 0 复制输入维度和 -1 自动推断维度的主路径。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op flatten --op reshape --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Tile` 与 `Concat` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`Tile` 的 CUDA 参考按输出坐标对输入维度取模，`Concat` 的 CUDA 参考按 concat axis 段落定位来源输入。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op tile --op concat --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

### 剩余风险

- 当前 numerical 是随机样本与固定 case 的默认门禁，不等价于 ONNX 每个 opset schema 的穷尽证明。
- QuantizeLinear/DequantizeLinear 的新版属性，例如 block quantization、output_dtype、precision、saturate 等，仍需结合目标 opset 再决定是否扩展。
- ROI、序列和谱算子已进入默认门禁，但仍建议后续继续补充更多 corner case，例如多方向序列、不同布局、异常轴、空张量和边界 window 参数。
