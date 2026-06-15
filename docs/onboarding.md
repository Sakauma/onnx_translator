<!--
/**
  ******************************************************************************
  * @file        onboarding.md
  * @author      Egor Izmaylov
  * @brief       记录工程接手、环境验证和门禁验收流程。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 工程接手指南

## 当前验收口径

本工程当前验收目标聚焦在三个方面：

1. 普通数值/张量算子的主路径必须由 C 后端承载，Python 只负责 ONNX 属性解析、shape 推导、ctypes 调度和必要的复杂语义 fallback。
2. 工程门禁必须能自动证明主路径覆盖没有回退，包括 C ABI 声明/实现一致、无 Python-only 普通数值主路径、已实现算子进入默认数值计划、当前 ONNX 最新默认域名称级导入无缺口。
3. 新同事接手后应能通过统一 CLI 完成环境检查、构建、pytest、图验证、CUDA verifier 编译和 numerical 正确性验证。

不属于当前强制验收的内容包括：所有官方属性组合穷尽证明、所有官方 dtype constraint 组合穷尽证明、低 bit/float4 官方 packed TensorProto 存储闭环，以及受当前 ONNX ReferenceEvaluator 限制的 FLOAT8E8M0 `QuantizeLinear output_dtype=24` 官方对照。

## 推荐环境

推荐使用已有 `egor` 虚拟环境：

```bash
export PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python
$PYTHON tools/health_check.py
```

如需运行完整 CUDA numerical 门禁，环境中还需要 `nvcc`。如果 `nvcc` 不在 `PATH`，可以通过 `NVCC=/path/to/nvcc` 指定。

## 一键门禁

有 CUDA 的环境运行完整门禁：

```bash
$PYTHON tools/verify_all.py --iterations 1
```

日常完整回归可把 `--iterations` 提高到默认值或更高：

```bash
$PYTHON tools/verify_all.py
```

无 CUDA 或只改导入/shape 逻辑时，先跑 CPU 门禁：

```bash
$PYTHON tools/verify_all.py --skip-cuda
```

`tools/verify_all.py` 默认会执行严格算子覆盖审计，相当于自动运行：

```bash
$PYTHON tools/audit_ops.py --strict
```

只有在排查审计脚本自身问题时才使用 `--skip-audit`。

## 当前应看到的关键数字

当前覆盖报告应保持以下核心数字：

- 当前安装 ONNX 最新默认域名称级覆盖：`200/200`。
- 普通数值/张量算子 Python-only 主路径：`0`。
- 默认 active numerical plan：`178` 个唯一算子名称，`723` 条默认计划。
- 默认 active numerical plan 混合精度覆盖：`488` 条计划。
- 最近完整 pytest 记录：`318 passed, 1 skipped`。
- 最近完整 CUDA verifier 编译记录：`178` 个 verifier 成功。
- 最近完整 numerical 记录：`723/723` 默认计划通过。

覆盖报告刷新命令：

```bash
$PYTHON tools/audit_ops.py --strict --output docs/reports/operator_coverage.md
```

## 常见开发位置

- Python 算子兼容入口：`nn/Operators.py`。
- Python 算子实际实现：`nn/operators/`。
- ONNX 导入兼容入口：`nn/ONNXImport.py`。
- ONNX node factory：`nn/importer/node_factories_*.py`。
- C 公共 ABI：`tensor_ops/tensor_ops.h`。
- C 内部工具：`tensor_ops/tensor_ops_internal.h`。
- C 后端实现：`tensor_ops/tensor_ops_*.c`。
- CUDA reference：`cuda/verify_<op>.cu`。
- numerical 计划：`tools/numerical/cli.py`。
- numerical runner 和参数打包：`tools/numerical/runner.py`。
- pytest 覆盖：`tests/test_operator_*.py`。

## 新增或修改算子的最低要求

普通数值/张量算子应同步完成以下内容：

1. Python 算子类中保留 shape 推导、输入整理和 ctypes 调度。
2. C 后端实现主体计算，并在 `tensor_ops/tensor_ops.h` 声明公共入口。
3. CUDA verifier 使用独立公式或独立坐标映射，不直接复制 Python fallback。
4. `tools/numerical/cli.py` 增加默认计划，必要时覆盖 float32、float16、bfloat16、float8 或量化路径。
5. pytest 覆盖 ONNX reference 语义、边界属性和导入器行为。
6. `tools/audit_ops.py --strict` 必须通过。

## 常见失败排查

- `tools/health_check.py` 缺模块：先确认是否使用了 `egor` 虚拟环境，再安装缺失依赖。
- `nvcc` 缺失：确认 CUDA 是否安装，或设置 `NVCC` 环境变量。
- strict audit 报 Python-only 主路径：检查算子 `forward()` 是否真实调用 C `<op>_forward`，不要只在 `__init__` 或 helper 中引用函数名。
- strict audit 报 C ABI 不一致：检查 `tensor_ops/tensor_ops.h` 与 `tensor_ops/tensor_ops_*.c` 的公共 forward 声明和实现是否一致；`static` 内部 helper 不需要写入公共头文件。
- numerical 失败：先用 `--op <op>` 缩小范围，再检查 runner 参数打包、CUDA verifier 输出 dtype、C 后端低精度写回和容差设置。
- 图验证失败：先重新生成模型，再检查 `nn/importer/` 的属性解析和 `forward_()` shape 推导。

## 生成物清理

以下内容不应提交：

- `cache/`
- `onnx_model/`
- `result/`
- `tensor_ops.so`
- `.pytest_cache/`
- `__pycache__/`
- `tmp_*.bin`

提交前可运行：

```bash
git clean -fdX
git status --short
```
