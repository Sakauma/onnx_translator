# 仓库协作指南

## 项目结构与模块组织

本仓库实现 ONNX 模型导入、图构建、C 后端算子执行，以及 CPU/CUDA 数值正确性验证流程。根目录只保留仓库级入口文件，例如 `README.md`、`AGENTS.md`、`Makefile` 和 `requirements.txt`。工程命令统一由 `tools/cli.py` 调度，具体命令实现位于 `tools/commands/`。`nn/ONNXImport.py` 与 `nn/Operators.py` 是兼容入口，实际导入逻辑位于 `nn/importer/`，算子实现位于 `nn/operators/`。C 后端位于 `tensor_ops/`，公共 ABI 声明在 `tensor_ops.h`，共享内部工具在 `tensor_ops_internal.h`，具体实现按领域拆分到 `tensor_ops_*.c`，编译产物为 `tensor_ops.so`。CUDA 参考验证程序位于 `cuda/`，命名格式为 `verify_<op>.cu`。生成的 CUDA 可执行文件写入 `cache/`；生成或示例 ONNX 产物位于 `onnx_model/` 和 `result/`。文档集中在 `docs/`，历史报告和覆盖报告位于 `docs/reports/`。

## 构建、测试与开发命令

- `make`：使用 GCC、OpenMP 和数学库从 `tensor_ops/*.c` 构建 `tensor_ops.so`。
- `make clean`：删除已编译的共享库。
- `python tools/cli.py compile-cuda`：使用 `nvcc` 编译所有 `cuda/*.cu` 验证程序到 `cache/`。
- `python tools/cli.py create-model` 或 `python tools/cli.py create-graph-model`：生成 ONNX 测试模型。
- `python tools/cli.py graph-logic`：验证 ONNX 图解析和算子连接关系。
- `python tools/cli.py numerical`：对比 C 算子输出与 CUDA 参考实现的数值结果。
- `python tools/cli.py verify-graph`：验证指定模型的解析和图构建流程。

## 编码风格与命名约定

Python 代码应保持简洁、可读，并延续现有风格：使用 4 空格缩进，函数和变量采用 `snake_case`，算子类使用清晰的 ONNX 风格 `PascalCase`。C 和 CUDA 函数使用 `snake_case`，前向计算接口命名为 `<op>_forward`。CUDA 验证文件统一命名为 `cuda/verify_<op>.cu`。新增算子时，应同步更新 `tensor_ops/tensor_ops.h`、对应 `tensor_ops/tensor_ops_*.c`、`nn/operators/`、`nn/importer/node_factories_*.py`、`tools/numerical/` 和对应 pytest 覆盖。

## 测试指南

本项目包含 `tests/` pytest 回归测试，同时保留脚本级验证。修改算子后，依次运行 `make`、`python -m pytest -q tests`，并在需要 CUDA 数值对比时运行 `python tools/cli.py compile-cuda` 和 `python tools/cli.py numerical`。修改导入器或图逻辑后，运行 `python tools/cli.py graph-logic`、`python tools/cli.py verify-graph` 或 `python tools/verify_all.py --skip-cuda`。新增算子测试应写入对应 `tests/test_operator_*.py` 文件，并在 `tools/numerical/cli.py` 中补充数值计划，覆盖形状推断、广播、类型提升，以及相关数值边界情况。

## 提交与 Pull Request 规范

近期提交历史使用简短的类型前缀，例如 `feat: ...`、`fix: ...` 和 `chore: ...`；提交信息应简洁、明确，并说明变更范围。Pull Request 应描述修改的算子或导入行为，列出已运行的验证命令，在有相关 Issue 时进行关联；仅当可视化输出变化时，才附带截图或生成的图文件。

## 安全与配置建议

除非明确需要，不要提交 `cache/*` 等生成的二进制文件。报告复现问题时，请注明本地 CUDA、GCC、Python、`numpy`、`torch` 和 `onnx` 版本，便于定位环境差异。
