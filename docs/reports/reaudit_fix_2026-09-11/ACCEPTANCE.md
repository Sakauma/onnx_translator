# ONNX translator 复审修复独立验收

## 验收对象

- 七项修复 source/test commit：`8a6f46985ad16265c5abc4283984f62ba98be2e1`
- CUDA QuantizeLinear follow-up commit：`b8d90098176f0bec35c1ddea334781697f9881b7`
- 审计基线：`f253f634dd251f348f432e62e318faf25198fd30`
- `tensor_ops.so` SHA-256：`602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`
- 固定执行环境：WSL `ubuntu2004`，Python `/home/sakauma/data/miniconda3/envs/egor/bin/python`，PATH 见各 command/meta 原始记录。

## 已独立核对的门禁

- Fresh CUDA 编译：正式第二轮 tool RC `0`，stdout footer 为 `compiled=178 skipped=0`，stderr 为空；编译后 178 个唯一源码与 178 个 executable 一一对应，无 missing/orphan。该项只证明 verifier 编译与库存，不宣称 178 个程序均已完成 GPU 数值动态执行。
- 最终全量 pytest：follow-up commit 上的 attempt2 原始 stdout 为 `537 passed, 1 skipped in 23.16s`，stderr 为空，RC `0`。唯一 skip 是 `tests/test_operator_activation_semantics.py:115`，原因为 `Celu does not support float16 in ONNX17`；没有 CUDA device、driver 或 executable 缺失类 skip。此前七项修复 commit 上的首轮结果为 `528 passed, 1 skipped`、RC `0`。
- `graph-logic`：RC `0`，stderr 为空；读取未跟踪模型 artifact `onnx_model/model.onnx`，SHA-256 `3da592ce6de1f81f0c93b1c5139a0bc0497c16bdfa3f527d4105a4f6fca27348`，解析 140 nodes、导入 148 operators 并完成 `Graph.forward_`。
- `verify-graph --no-clean`：RC `0`，stderr 为空；使用同一模型与默认 strict importer，导入 148 operators，无 `GenericNode`，完成声明输出校验和 `Graph.forward_`。

## Numerical gate

状态：**PASS**。

最终 attempt2 使用 `/home/sakauma/data/miniconda3/envs/egor/bin/python -u tools/cli.py numerical --iterations 3 --skip-plots`，从 `2026-09-11T04:27:35Z` 运行至 `2026-09-11T04:39:44Z`。meta 精确记录 source SHA `b8d90098176f0bec35c1ddea334781697f9881b7`、共享库 SHA `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66` 和 178-verifier inventory；stderr 为空，真实 Python/wrapper RC 为 `0`。

对完整原始 stdout 独立计数得到 723 个 `Testing` plan 标题和 723 个 `Pass (3/3)`，即 2169 次 plan iterations；失败、crash、traceback、未完成 plan 与基础设施异常均为 0。首轮失败的默认 FLOAT16/FLOAT16/INT8 和 BFLOAT16/BFLOAT16/INT8 QuantizeLinear plans 在 attempt2 中分别明确记录 `Pass (3/3)`。因此 numerical 的动态 GPU 结论来自 723 个实际 plan 日志，不是从 178 个 executable 的编译库存外推。

首次完整运行已自然结束为 723 started / 721 pass / 2 fail / 0 exception / RC `1`。两项失败均为 QuantizeLinear 默认 FLOAT16/BFLOAT16 precision 的旧 CUDA oracle 提升到 float32；原始失败证据继续保留，没有重分类或覆盖。follow-up commit 仅修改 `cuda/verify_quantize_linear.cu`、QuantizeLinear 的 CUDA params 编码和新增真实 GPU 回归；两次 numerical inventory 对比也只有 `cache/verify_quantize_linear` 的 SHA 改变，从 `e10466f57610035a968b12836382f00c381ec2ff4c7d0743560fa5e4de9c9be3` 变为 `0c12b072e454d8e80aeefa3ac9af1ea3f7a9d03850a69e67bc0b849d2c7bf8be`，其余 177 个 verifier 不变。

## 最终验收范围

本轮 PASS 只覆盖 REAUD-001 至 REAUD-007、为 QuantizeLinear 修复所需的 half min-subnormal codec 边界，以及随后暴露的 CUDA QuantizeLinear precision oracle 协议。DynamicQuantizeLinear 全零输入继续采用既有 `scale=1` 行为；该外部规范/实现歧义没有在本轮解决。本报告不声称仓库不存在其他 bug，也不声称运行了 `verify_all.py` 或未列出的环境与平台门禁。
