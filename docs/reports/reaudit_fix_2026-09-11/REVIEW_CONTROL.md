# REAUD-003..006 控制流修复独立复核

## 结论

只读复核 `nn/operators/common.py`、`nn/operators/sequence_optional_control.py` 与 `tests/test_reaudit_control_boundaries.py` 后，未发现阻断或需要追加修改的问题。负责 agent 报告固定环境定向测试为 RC `0`、`23 passed, 0 skipped`；本复核没有重复执行该测试。

## 逐项检查

- REAUD-003：Loop zero-trip 从 body tensor inputs 与实际 runtime `Tensor`/`Tensor_` shape 建立符号绑定。绑定过程只读 shape 元数据，不把 `Tensor_` 转为零数组；scan element shape 的未绑定维、缺失 tensor shape 和不支持 dtype 均明确抛错。空 scan 本身只分配零元素 ndarray，shape 为 `(0, *resolved_element_shape)`。
- REAUD-004：Loop carried state 进入 ReferenceEvaluator 时递归保留 sequence 容器，迭代 body 输出不再无条件 `np.asarray`；最终值按 body output `TypeProto` 递归恢复 tensor、sequence 和 optional kind。zero-trip 与 one-trip sequence 回归都经过外层 `SequenceLength` 检查，覆盖容器保持而不只检查 Python 类型。
- REAUD-005：SequenceMap 在迭代前按 `len(body.output)` 初始化 buckets，空输入仍返回每个声明输出对应的 empty sequence；非空路径检查 body output arity 并按声明类型转换。专属测试覆盖空/非空和两个输出。
- REAUD-006：Scan runtime 非空堆叠、empty-output 构造和 `forward_` shape-only 都以最终 output rank 调用同一 `_normalized_axis`；`-1`、最低合法负轴和正负越界均有回归。input axis 在 runtime 与 shape-only 中也使用同一规范化规则。

## 范围与限制

- 本轮 confirmed finding 只动态要求 Sequence carried state；Optional 的递归转换虽已实现，但没有把 Optional 宣称为本轮动态验收项。
- `Loop.forward_` 对未知符号维仍沿用通用占位 shape 行为，这不参与 REAUD-003 的 runtime zero-trip empty scan 路径；该路径现在对未知维 fail closed。
