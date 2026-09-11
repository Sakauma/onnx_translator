# DynamicQuantizeLinear 全零行为修复总结

源提交 `a1d4940cc694ac65fbb77a993b7b800fcafd6ee5` 已将有限、非空且值级全零的 float32 输入（包括混合 `+0.0/-0.0`）对齐 ONNX 1.21 `ReferenceEvaluator`：输出 `y` 为同形状 `uint8` 全零，`y_scale` 为 float32 `1/255`，`y_zero_point` 为标量 `uint8(0)`。这是项目兼容策略，不表述为 opset 17 的规范唯一要求。

C 后端、CUDA verifier、numerical 固定计划、输入生成及特殊输出 oracle 已同步更新。oracle 根据每轮实际输入自动识别此边界，并分别校验 C 与 CUDA，避免二者一致产生错误结果时假通过。非零常量、普通非退化输入和非零 subnormal range 下溢路径保持原有行为；empty、NaN 与 infinity 支持范围未改变。

验证全部通过：fresh `make` 和 DQL CUDA 单目标编译 RC 0；新 Reference/C/GPU 回归 11 passed；既有精度回归 11 passed；misc DQL 回归 3 passed；harness 门禁 80 passed；完整 pytest 为 561 passed、1 个既有 Celu float16 skip；完整 numerical 为 725/725 个计划、2175/2175 次迭代通过，Python 与 wrapper RC 均为 0，stderr 空。详细证据见 [VALIDATION.md](VALIDATION.md)，实现说明见 [IMPLEMENTATION.md](IMPLEMENTATION.md)，策略说明见仓库 `docs/dynamic_quantize_zero_policy.md`。

主工作区 `main` 随后安全 fast-forward 到源提交并完成 fresh 构建和定向复验：C 库哈希与隔离构建一致；CUDA verifier 的唯一差异是 `.strtab` 内 nvcc 临时文件进程编号；主目录真实 Reference/C/GPU 测试 11 passed，默认 DQL numerical 3 plans × 3 iterations 全部通过。同步证据见 [MAIN_SYNC.md](MAIN_SYNC.md)。
