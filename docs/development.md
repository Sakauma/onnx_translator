<!--
/**
  ******************************************************************************
  * @file        development.md
  * @author      Egor Izmaylov
  * @brief       记录新增算子的开发检查清单和验证步骤。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 算子开发标准流程清单

开发一个新算子通常需要修改以下 **5 个区域** 的代码：

核心原则：普通数值/张量算子的主体计算必须优先放到 `tensor_ops/tensor_ops_*.c`，Python 只承担 ONNX 导入、参数整理、shape 推导、ctypes 调度，以及确实难以静态表达的动态 fallback。不要把新增算子的主要功能长期用 Python 实现，否则无法体现本工程的 C 后端和 CUDA 验证价值。

当前 ONNXImport 覆盖的普通数值/张量算子均应具备 C runtime path。复杂算子可以保留 Python fallback 作为对照或动态语义兜底，但不能只实现 Python runtime。

发布级目标要求实现持续和当前安装的 ONNX 默认 domain 最新官方算子对齐。新增官方算子时，除非属于纯图控制、序列、字符串或元数据调度类，否则主功能必须进入 C 后端，并同步进入 CUDA verifier 与 active numerical plan。

---

## 1. C 后端接口 (`tensor_ops/tensor_ops.h`)
这是 Python 与 C 交互的接口。

* **定义参数结构体**：如果算子参数超过 3-4 个（如 Conv/Pool），建议定义一个 `struct` 来打包参数。
* **声明前向函数**：`void xxxx_forward(...)`。

## 2. C 后端实现 (`tensor_ops/tensor_ops_*.c`)
这是核心计算逻辑，**精度优先**。

* **选择分域文件**：按算子职责放入 `tensor_ops_elementwise.c`、`tensor_ops_activation_extra.c`、`tensor_ops_shape_index.c`、`tensor_ops_shape_grid.c`、`tensor_ops_conv_pool_roi.c`、`tensor_ops_pool_roi.c`、`tensor_ops_global_pool.c` 等文件。
* **复用辅助函数**：共享 dtype、Tensor 读写、坐标和宏模板逻辑优先放在 `tensor_ops/tensor_ops_internal.h`。
* **核心逻辑**：
    * 解析输入形状和参数。
    * 使用多重循环遍历输出 Tensor 的每个元素。
    * **计算逻辑**：使用 `double` 进行高精度中间计算。
    * **并行化**：在最外层循环加上 `#pragma omp parallel for`。
* **边界处理**：注意处理 Padding、Stride 以及可能的越界情况。

## 3. Python 算子封装 (`nn/operators/`)
这是 NPS 前端定义。

* **定义 ctypes 结构**：如果 C 端定义了 struct，这里要定义对应的 `ctypes.Structure`。
* **新建算子类**：在对应职责模块中继承 `Ops`；`nn/Operators.py` 只保留兼容 re-export。
    * `__init__`：保存参数，并设置 `self.lib.new_op_forward.argtypes`（**这一步很容易忘！**）。
    * `forward` (真实计算)：
        1.  推断输出形状 `out_shape`。
        2.  使用 `_numpy_to_ctensor` 转换输入。
        3.  打包参数（如果是 struct）。
        4.  调用 `self.lib.new_op_forward`。
        5.  使用 `_ctensor_to_numpy` 转回结果并释放 C Tensor 内存。
    * `forward_` (形状推断)：仅复制 `forward` 中的 `out_shape` 计算逻辑，返回占位符 `Tensor_`。

## 4. ONNX 解析适配 (`nn/importer/`)
这是模型导入逻辑。

* 在 `nn/importer/node_factories_*.py` 中添加 `@register_factory("NewOp")`。
* **提取属性**：遍历 `node.attribute`，提取 `pads`, `strides`, `axis` 等参数。
* **实例化**：调用 `nn.Operators.NewOp(...)` 将节点加入图列表。

## 5. 验证 (`cuda/` & `tools/numerical/`)
确保结果正确。

* **编写 CUDA 真值 (`cuda/verify_new_op.cu`)**：
    * 使用 `double` 编写 Kernel，逻辑要求标准（不追求速度，只求对）。
    * 在 `main` 函数中处理参数读取（如果是复杂算子，需读取 `params.bin`）。
    * **记住一定使用 `malloc` 和 `free`！**
* **更新验证脚本 (`tools/numerical/`)**：
    1.  `import` 新算子类。
    2.  在 `tools/numerical/runner.py` 中添加该算子的 **参数打包逻辑**（将 Python 参数转为 bytes 传给 CUDA）。
    3.  在 `tools/numerical/cli.py` 的默认计划中添加测试用例（覆盖 float32, float16, float8 等混合精度场景）。

## 6. 覆盖审计

完成算子修改后刷新审计报告：

```bash
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python audit
```

报告会写入 `docs/reports/operator_coverage.md`，用于确认 `forward()` 是否真正走到 C runtime path、是否只有合理的 Python 调度/元数据算子留在 Python，以及 CUDA verifier / active numerical plan 是否同步更新。

## 7. 发布级补充门禁

涉及 C 后端、算子覆盖、包入口或性能敏感路径时，还应运行：

```bash
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python abi-check
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python onnx-semantic-matrix
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python release-check
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python release-preflight
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python release-artifacts
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python manylinux-wheels
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python manylinux-wheelhouse-check
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python manylinux-wheels-full
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python manylinux-wheelhouse-check-full
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python model-smoke
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python benchmark-smoke
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python benchmark-smoke-report
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python benchmark-baseline-check
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python benchmark-fixed-runner-check
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python benchmark
make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python sanitize
```

`onnx-semantic-matrix` 会刷新官方 ONNX latest 默认域语义矩阵，要求每个官方算子都有 import、operator class 或 deprecated canonical alias，以及 numerical/CUDA/pytest/orchestration 等强证据；`release-check` 是快速发布就绪审计；`release-preflight` 是发布前最后一轮聚合门禁，会串联语义矩阵、release、测试、模型、性能、包构建和 sanitizer 并输出 `result/release_preflight.json`；`release-artifacts` 验证 sdist/wheel、metadata、sdist 源码完整性和从 sdist 重建 wheel 的安装 smoke；`manylinux-wheels` 使用 Docker/cibuildwheel 构建 Linux 发布 wheel，`manylinux-wheelhouse-check` 检查 wheel 是否包含 C runtime、是否是非纯 Python 平台 wheel、platform tag 是否为 manylinux，并拒绝 `.data/purelib` 中的共享库；`manylinux-wheels-full` 和 `manylinux-wheelhouse-check-full` 用于发布验收，强制证明 `cp310/cp311/cp312` 完整矩阵；`model-smoke` 用 CNN、Transformer block 和 Embedding MLP 代表性模型验证导入、图推导和 ONNX reference 数值对齐；`benchmark-smoke` 是适合 CI/PR 的保守性能门禁；`benchmark-smoke-report` 会生成可上传的 `result/benchmark_smoke.json`；`benchmark-baseline-check` 使用 `docs/performance_baseline.json` 做版本化回归检查并生成 `result/benchmark_baseline_check.json`；`benchmark-fixed-runner-check` 使用 `docs/performance_fixed_runner_baseline.json` 和固定 `PERF_RUNNER_ID` 做更严格的固定机器性能回归检查；`benchmark` 记录核心 C 后端路径的性能基线；`sanitize` 用 ASan/UBSan 运行 C 后端回归子集和代表性模型组合路径，用于尽早发现越界访问、use-after-free 和未定义行为。

如果修改了 `tensor_ops/tensor_ops.h` 的公开枚举、结构体或函数签名，必须明确评估兼容性；确认这是有意 ABI 变化后，使用 `$PYTHON tools/abi_manifest.py --write` 刷新 `docs/abi_manifest.json`。

---
