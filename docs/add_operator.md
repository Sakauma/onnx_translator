<!--
/**
  ******************************************************************************
  * @file        add_operator.md
  * @author      Egor Izmaylov
  * @brief       说明新增或补齐 ONNX 算子的推荐开发流程和验证清单。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 新增算子流程

新增算子时，优先保证公共导入路径、C ABI 和验证入口稳定。除控制流、序列、文本等特殊语义外，普通数值算子的主体计算应落在 C 后端。

## 1. Python 算子类

1. 在 `nn/operators/` 中选择对应职责模块，新增算子类。
2. 共享 dtype、broadcast、shape 或采样逻辑优先放到 `nn/operators/common.py`。
3. 保留 `forward_()` 形状推断路径；数值 `forward()` 应优先调用 C 后端，必要时保留 Python fallback。
4. 确认 `nn/operators/__init__.py` 能 re-export 新类，从而保持 `from nn.Operators import <Op>` 可用。

## 2. ONNX 导入工厂

1. 在 `nn/importer/node_factories_*.py` 中新增或调整 factory。
2. 使用 `@register_factory("<OpType>")` 注册 ONNX op_type。
3. 从 `ImportContext.get_dtype()` 查询输出 dtype，避免重新推断整图。
4. 对 optional 输入保留空字符串占位，不要随意压缩输入位置。

## 3. C ABI 与实现

1. 如果需要新的 C 入口，先在 `tensor_ops/tensor_ops.h` 增加声明。
2. 将实现放入对应 `tensor_ops/tensor_ops_*.c` 文件。
3. 共享内部逻辑放入 `tensor_ops/tensor_ops_internal.h`，不要复制到多个 `.c`。
4. 运行 `make PYTHON=$PYTHON check`，确认 C 后端可编译且 Python 入口可编译。

## 4. CUDA verifier

1. 新增 `cuda/verify_<op>.cu`，命名与数值计划中的 `op_name` 保持一致。
2. 运行 `python tools/cli.py compile-cuda`，确认可执行文件写入 `cache/verify_<op>`。
3. CUDA 输出使用二进制文件协议，与 `tools/numerical/cuda.py` 保持一致。

## 5. 数值计划

1. 在 `tools/numerical/cli.py` 的 `build_default_plans()` 中加入计划。
2. 覆盖基础形状、广播、类型提升和主要边界值。
3. 对复杂参数使用 `init_args`，并在 `tools/numerical/runner.py` 中补充参数打包逻辑。
4. 用 `python tools/cli.py numerical --op <op> --iterations 5 --skip-plots` 做单算子验证。

## 6. Pytest 覆盖

1. 按算子域选择 `tests/test_operator_*.py` 文件新增回归测试。
2. 需要屏蔽 C 后端时使用 `conftest.py` 中的 `_disable_c_backend()`。
3. 覆盖 ONNXImport 属性解析、`forward_()` 形状推断、Python fallback 和 C 后端路径。
4. 完成后运行 `python -m pytest -q tests`。

## 最小验收清单

```bash
make PYTHON=$PYTHON check
$PYTHON -m pytest -q tests
$PYTHON tools/cli.py compile-cuda
$PYTHON tools/cli.py numerical --op <op> --iterations 5 --skip-plots
$PYTHON tools/audit_ops.py --output /tmp/onnx_translator_audit.md
```
