<!--
/**
  ******************************************************************************
  * @file        README.md
  * @author      Egor Izmaylov
  * @brief       汇总 docs 目录下的工程说明、开发流程和验证报告入口。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 文档索引

本目录集中存放工程说明、开发流程和验证报告。根目录只保留稳定命令入口、构建文件、依赖文件、README 和 AGENTS 指令。

## 工程说明

- [工程架构说明](architecture.md)：说明 ONNX 导入、算子层、C 后端、CUDA verifier 和测试框架的关系。
- [工程接手指南](onboarding.md)：说明环境检查、一键门禁、当前关键覆盖数字和常见失败排查。
- [新增算子流程](add_operator.md)：说明新增算子的 Python、ONNX factory、C ABI、CUDA verifier、数值计划和 pytest 覆盖步骤。
- [开发流程清单](development.md)：保留较细的算子开发检查清单。

## 报告

- [验证总结](reports/verify_summary.md)：记录当前数值验证和图结构验证覆盖范围。
- [算子覆盖报告](reports/operator_coverage.md)：由 `make audit` 或 `tools/audit_ops.py --strict` 生成，记录算子覆盖状态。

## 根目录保留项

`AGENTS.md` 保留在仓库根目录，方便 Codex 和其他代理工具自动读取仓库协作规则。工程工具命令集中到 `tools/cli.py` 和 `tools/commands/`，根目录不再放散落的命令脚本。
