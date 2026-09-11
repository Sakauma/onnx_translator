# 主检出目录集成验证

日期：2026-09-11

主目录：`/mnt/d/workspace/onnx_translator`

分支：`main`
集成提交：`e0a03c6b3e71ba5c9562624f783c3ac6e4165322`

## 快进前置检查

主目录初始 HEAD 为基线 `96d070e6f1a26f61ee0cf727c9616148b8680629`。tracked working tree 和 index 均为空；原有未跟踪报告完整保留，且本轮 `docs/reports/post_push_fixes_2026-09-11/` 路径不存在、没有碰撞。Git 确认基线是修复提交的祖先后，`git merge --ff-only e0a03c6b3e71ba5c9562624f783c3ac6e4165322` 成功，主目录 HEAD 更新到该提交。

## 构建与产物身份

固定使用 WSL `ubuntu2004`、egor Python 和项目规定 PATH。主目录只执行一次 `make`，随后只对 LayerNormalization verifier 执行一次 `compile-cuda --op layer_normalization --force`，未重编其他 177 个 verifier。

| 门禁 | 结果 |
| --- | --- |
| `make` | PASS，RC 0，stderr 为空 |
| `tensor_ops.so` SHA-256 | `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89`，与隔离工作树最终产物一致 |
| LayerNormalization CUDA 强制重编 | PASS，RC 0，`compiled=1 skipped=0`，stderr 为空 |
| 主目录 LayerNormalization verifier SHA-256 | `2f0dca3994e5ed6f673778e4fa0c6caca2cb9e8d90828740384b31ecf20641ae` |
| verifier 数量 | 178 |

主目录 verifier 哈希与隔离工作树的 `cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074` 不同。本轮证据确认它由已验收源码在预期编译环境中全新生成，且真实 GPU 定向门禁通过；未对哈希差异指定原因，也没有为追求相同哈希重复构建。

## 主目录定向验证

| 门禁 | 结果 |
| --- | --- |
| `pytest -q tests/test_post_push_shape_semantics.py tests/test_post_push_layernorm_semantics.py` | PASS，20 passed in 4.00s，RC 0，stderr 为空 |
| LayerNormalization numerical，3 次迭代 | PASS，9 个计划、27/27 次，RC 0，报告绝对/相对误差为 0，stderr 为空 |
| Resize numerical，3 次迭代 | PASS，5 个计划、15/15 次，RC 0，报告绝对/相对误差为 0，stderr 为空 |

各命令、stdout、stderr、RC 和身份元数据位于 `evidence/main/`。所有阶段均为首次运行通过，没有重跑完整 pytest、完整 numerical 或其他 177 个 CUDA verifier，也没有修改旧模型和结果目录。

## 状态

主目录源码快进、构建和指定定向门禁均为 **PASS**。本阶段未 push，也未提交本轮报告。
