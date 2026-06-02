<!--
/**
  ******************************************************************************
  * @file        README.md
  * @author      Egor Izmaylov
  * @brief       记录 ONNX Translator 相关说明。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# ONNX Translator

本项目实现 ONNX 模型导入、图构建、C 后端算子执行，以及 CUDA 参考程序数值验证。

## 文档入口

- [工程架构说明](docs/architecture.md)：说明 ONNXImport、算子层、C 后端、CUDA verifier 和测试框架的关系。
- [新增算子流程](docs/add_operator.md)：说明新增算子的 Python、ONNX factory、C ABI、CUDA verifier、数值计划和 pytest 覆盖步骤。
- [开发流程清单](docs/development.md)：记录算子开发时需要同步修改和验证的位置。
- [验证总结](docs/reports/verify_summary.md)：记录当前数值验证和图结构验证覆盖范围。
- [算子覆盖报告](docs/reports/operator_coverage.md)：记录 ONNXImport、C 后端、CUDA verifier 和数值计划覆盖情况。

## 后端化边界

普通数值/张量算子的核心计算应优先落在 `tensor_ops/tensor_ops_*.c`，Python 侧主要负责 ONNX 属性解析、shape 推导、ctypes 调度和少量动态语义 fallback。新增或补齐算子时，不应把主体计算长期停留在 `nn/operators/`。

当前 ONNXImport 覆盖的普通数值/张量算子均应具备 C runtime path；审计报告中不应出现 Python-only 待后端化项。复杂算子仍可在 Python 保留等价 fallback，但不能作为唯一运行路径。

## 环境

推荐在 WSL/Linux 中运行。当前验证使用已有环境：

```bash
PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python
```

安装依赖：

```bash
$PYTHON -m pip install -r requirements.txt
```

如果只做导入和图构建验证，可以暂时不安装 CUDA；如果要跑完整数值正确性验证，需要 WSL/Linux 中可用的 `nvcc`。

## 构建与静态检查

环境健康检查：

```bash
$PYTHON tools/health_check.py
```

```bash
make clean
make PYTHON=$PYTHON check
```

`make check` 会编译 `tensor_ops.so`，并对主要 Python 入口做 `py_compile` 检查。当前目标是保持 `-Wall -Wextra` 下无 C 编译警告。

## 常用验证命令

```bash
make PYTHON=$PYTHON check
$PYTHON -m pytest -q tests
$PYTHON tools/cli.py compile-cuda
$PYTHON tools/cli.py numerical --iterations 20 --skip-plots
$PYTHON tools/verify_all.py --skip-cuda
$PYTHON tools/audit_ops.py --output /tmp/onnx_translator_audit.md
```

如果只改 Python 导入或 shape 逻辑，优先运行 `make PYTHON=$PYTHON check` 和 `$PYTHON -m pytest -q tests`。如果改 C 后端或 CUDA verifier，应补充 `$PYTHON tools/cli.py compile-cuda` 和对应 `$PYTHON tools/cli.py numerical --op <op>`。

## 推荐开发流程

1. 修改前运行 `tools/audit_ops.py` 或目标测试，确认当前覆盖基线。
2. Python 算子改动放在 `nn/operators/`，旧入口 `nn/Operators.py` 只做兼容 re-export。
3. ONNX 属性解析放在 `nn/importer/node_factories_*.py`，通过注册表接入。
4. 普通数值算子的核心计算放在 `tensor_ops/tensor_ops_*.c`，共享辅助逻辑放在 `tensor_ops/tensor_ops_internal.h`。
5. 数值验证计划放在 `tools/numerical/cli.py`，测试按领域放进 `tests/test_operator_*.py`。
6. 提交前至少运行 `make PYTHON=$PYTHON check`、`$PYTHON -m pytest -q tests` 和 `tools/audit_ops.py`。

## 一键验证门禁

完整本地门禁会清理旧产物、检查环境、编译 CUDA verifier、构建 C 后端、运行单元测试、生成/验证两个 ONNX 模型，并执行 CUDA 数值正确性验证：

```bash
$PYTHON tools/verify_all.py
```

无 CUDA 的环境可以跑 CPU 门禁：

```bash
$PYTHON tools/verify_all.py --skip-cuda
```

也可以通过 Makefile 调用：

```bash
make PYTHON=$PYTHON verify-cpu
make PYTHON=$PYTHON verify
```

调试时保留生成物：

```bash
$PYTHON tools/verify_all.py --keep-artifacts
```

## 覆盖审计

刷新算子实现统计报告：

```bash
make PYTHON=$PYTHON audit
```

报告写入 `docs/reports/operator_coverage.md`，会区分实际 C runtime path、合理 Python 调度/元数据算子、当前暂缓后端化算子、当前暂缓深度语义/数值验证算子，以及 CUDA/数值验证覆盖。

## 生成 ONNX 模型

使用 PyTorch 导出综合算子模型：

```bash
$PYTHON tools/cli.py create-model
```

生成只覆盖图结构/shape 类算子的模型：

```bash
$PYTHON tools/cli.py create-graph-model
```

模型默认写入 `onnx_model/model.onnx`。

## 图导入和构建验证

严格验证模型导入、图连接、shape 推导和可视化生成：

```bash
$PYTHON tools/cli.py verify-graph --model ./onnx_model/model.onnx --task-name nps_verification
```

默认 strict 模式会在不支持或解析失败的节点上直接失败。只有在明确接受占位节点时才使用：

```bash
$PYTHON tools/cli.py verify-graph --model ./onnx_model/model.onnx --no-strict --allow-generic
```

## CUDA 数值验证

先编译 CUDA 参考程序：

```bash
$PYTHON tools/cli.py compile-cuda
```

可通过环境变量覆盖路径：

```bash
CUDA_DIR=cuda CACHE_DIR=cache NVCC=/path/to/nvcc $PYTHON tools/cli.py compile-cuda
```

运行数值验证：

```bash
$PYTHON tools/cli.py numerical --iterations 20 --skip-plots
```

只跑某个算子：

```bash
$PYTHON tools/cli.py numerical --op add --iterations 5 --skip-plots
```

如果 `cache/verify_<op>` 缺失，数值验证会返回非零退出码。

## 单元测试

```bash
$PYTHON -m pytest -q
```

当前测试覆盖图执行重复调用、图输出返回、ONNXImport strict 失败、GenericNode 降级记录，以及重点算子的导入、shape 推导、Python fallback 和 C 后端路径一致性。
