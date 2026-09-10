# Bug 修复独立静态验收（2026-09-10）

审查范围：基线 `42f2527347d1c02b17395e90c08b0ee7a46b2eaf` 至已审快照 `4ba7f76e5608419e0b6df4cd77b2f21d48d03916` 的 NUM / IMP / GRAPH / GATE / CUDA 已提交变更。未把 worktree 中未提交内容计入结论。按任务约束未运行 Python、构建或测试。

## 缺陷关闭状态

### 已关闭：CUDA ReduceLogSumExp 非有限 oracle

- `7360021` 在 kernel 中显式处理 NaN、任一 `+Inf` 和全 `-Inf`，并保留有限值的稳定 max-shift 公式。
- 回归覆盖 `[+Inf]`、`[+Inf,-Inf]`、全 `-Inf`、NaN 优先及大有限值；验证方报告真实 GPU 5 项通过。原 P2 关闭。

### 已关闭：多 sidecar 中途失败遗留文件

- `a939823` 引入 `_read_sidecar_group`，用组级 `finally` 清理所有成员，并迁移各多输出协议。
- 回归精确覆盖“第一个合法、第二个 malformed、第三个仍存在”，验证方报告 runner 定向 16 项通过。原 P2 关闭。

### 已关闭：CUDA_CHECK 机械嵌套导致编译失败

- `0c05a40` 曾在 34 个 verifier 中把同一行多个调用错误嵌入一个 `CUDA_CHECK`，例如 `verify_abs.cu:36`；该提交不能通过 nvcc。
- `4ba7f76` 将这些调用逐一拆成独立宏调用。复核原两类嵌套模式均为零，条件释放和调用顺序保持不变。174 个 kernel launch 对应 174 个调用点；文本统计的第 175 个 `CUDA_CHECK_LAUNCH` 是公共宏定义本身。
- P1 关闭。`final_compile_cuda_all.log:180` 进一步记录真实全编译 RC 0，`compiled=178 skipped=0`。

## 覆盖结论

- IMP：dispatch key 包含 canonical domain，大小写兼容键也只在同域内生成，未发现同名跨域 fallback。schema capability 用 anchor 对应的 effective `since_version` 精确匹配，future opset 和未实现 revision 会拒绝；现有 overlay 保留已登记的 18/20/23/26 模型。Softmax 11 在真实 forward 中规范化并校验 axis，按 `[0, axis)` / `[axis, rank)` flatten 为二维，计算后恢复原 shape；默认 axis=1、显式负轴、新版默认末轴均有回归。
- IMP external initializer：官方 `numpy_helper.to_array(..., base_dir=model_parent)` 在导入边界解析旁置数据；strict 错误包含 initializer、模型路径和原异常，non-strict 生成不可执行 `GenericNode` 诊断。普通内嵌 initializer 的原路径仍共用该转换。
- GRAPH：声明输出名称和顺序传给 `Graph`，只收集声明输出；校验数量、dtype、rank、已知维，未知 rank 保留为 `None`，符号/未知维不被伪造为 1。现有测试覆盖中间声明输出、额外终端、顺序及 mismatch。
- NUM-001：C 与 CUDA 分别改用 ties-to-even；`[-253,257,0]` 的测试使用精确整数期望并通过调用计数证明真实 C 入口。计划明确留白的 `[-1,0,1]` float32 scale 与 C double 中间精度差异未被本批解决，最终报告不得扩大声明。
- NUM-002：C、Python fallback 与 CUDA oracle 均显式处理 NaN、`+Inf`、全 `-Inf`；C 测试证明真实 C 路径，CUDA 固定反例由真实 GPU 验证。
- NUM-003：构造器 axes 与 runtime axes 在同一规范化逻辑汇合；空 axes 的 noop=0 全归约、noop=1 恒等，负轴、keepdims、shape-only 与 fallback 均有针对性测试。
- GATE：Slice helper 导入恢复；每个 plan 的异常边界会记录失败并继续后续 inventory，汇总存在失败时最终 RC=1。TopK 与 special sidecar 校验 dtype 对应的精确字节数，缺失、截短和多余字节均拒绝，组级失败会清理整组文件。
- CUDA-001：全部 verifier 可直接或经共享头到达 `verify_common.cuh`；公共 helper 对 runtime、launch、同步及文件 I/O 失败输出 stderr 并非零退出。静态复核未见裸 runtime 调用，launch/check 数对应。动态 shape 的读取仍保留原显式返回值检查，没有把可选 EOF 错当成必需输入。

## 最终动态验收状态

- CPU 全门禁已通过：`final_verify_all_skip_cuda_retry.log:54` 记录 `437 passed, 10 skipped`，日志末尾记录模型导入、图形状推断和可视化成功；命令 RC 0。
- CUDA 178 全编译已通过：`final_compile_cuda_all.log:180` 记录 `compiled=178 skipped=0`；命令 RC 0。
- CUDA protocol 已通过：`final_cuda_protocol.log:2` 记录 `34 passed`、`0 skipped`；命令 RC 0。它包含真实 GPU 的 DynamicQuant 精确 half-even 和 ReduceLogSumExp 非有限反例。
- 原生 full numerical 已通过：`final_numerical_full.log` 记录 723 个 live plan 全部 `Pass (3/3)`，共 2169 次迭代比较、0 个失败标记，最后三项均为 inventory 末尾的 LSTM 计划；命令 RC 0。

## 验收中修复

动态验证发现未知 custom domain 被过早分类为 future opset。`937fde2` 将已导入域判定移到已知域 opset 上界检查之前：未知域返回 `unimported domain`；opset<=0、已知域 future opset 和 schema anchor 边界保持不变。

既有九个 importer 失败来自 graph/test 名明确标为 ONNX 17 的组合 fixture 遗漏 `opset_imports`，导致新版 ONNX helper 默认生成 opset 26 模型；后续逐节点暴露也证明这是模型级元数据问题。`6900d77` 将这些 fixture 显式固定为 opset 17，并撤销无依据添加的九个 opset 26 anchors。GridSample fixture 使用旧 schema 的 `bilinear`/`bicubic` 属性值，尤其不能作为现代 schema 支持证据；Reshape `allowzero` 在 opset 17 已合法。最终现代 opset 26 支持范围未被这些遗留 fixture 扩张。

## 仍需验证与范围限制

1. DynamicQuant `[-1,0,1]` 的 float32 schema/reference 计算链与 C double 中间值专项定性，仍是计划明确保留项。
2. ABI、自定义 allocator、错误 shape 缓冲区安全及 ASan/UBSan 不在本轮修复和验收范围；本报告不声称这些项目已验证或解决。
