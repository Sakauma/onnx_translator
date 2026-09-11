# REAUD-001/002 数值修复记录

## 设计与修改

- `DynamicQuantizeLinear` 的 FLOAT 路径现在按 ONNX 函数体的 dtype 顺序，在 float32 中物化 extrema、range、scale、zero-point 比值和逐元素 quotient，再执行 ties-to-even 舍入。全零 range 继续返回既有的 `scale=1`，没有选择复审记录中的规范歧义分支。
- `QuantizeLinear` 将除法精度解析为“显式 `precision` 优先，否则使用 scale dtype”。除法的两个操作数先转换到所选精度；FLOAT16 与 BFLOAT16 随后还通过现有 nearest-even codec 物化 quotient，FLOAT 与 DOUBLE 分别保留 float32、float64 运算。
- `float_to_float16` 的 half min-subnormal midpoint 边界允许 `shift==24` 进入 guard/sticky/LSB 舍入：精确 `±2^-25` ties-to-even 为 signed zero，刚越过 midpoint 的 float32 相邻值舍入为 `±0x0001`。
- 未修改 ABI、axis/block 参数展开或 importer 支持范围。

## 回归覆盖

`tests/test_reaudit_quantization_precision.py` 覆盖：

- DQL 精确复现值 `y=[0,4,2,255]`、scale bits `0x5c877e76`、zero point `2`；
- DQL 常规输入、对称 float32 最小正规数和保留的全零行为；
- QuantizeLinear FLOAT16 精确 bits fixture，默认/FLOAT16 得到 `30`，显式 FLOAT/DOUBLE 得到 `29`；
- float32 输入显式指定 FLOAT16 时，先转换两个操作数再除法；
- FLOAT16 最小 subnormal 的正负 midpoint tie 与相邻 float32 边界；
- FLOAT16 正负 ties-to-even 和 INT8 饱和边界。

## 验证状态

统一调度的 `make` 已实际运行并以 RC `0` 完成，详细环境、完整编译命令、产物时间与 SHA-256 见 `NUMERIC_BUILD.md`。

固定 WSL/Python 环境中的实际定向测试结果：

- `python -m pytest -q tests/test_reaudit_quantization_precision.py`：RC `0`，`11 passed in 1.18s`；
- `python -m pytest -q tests/test_operator_misc_semantics.py -k dynamic_quantize_linear`：RC `0`，`3 passed, 21 deselected in 1.03s`；
- `python -m pytest -q tests/test_operator_core_numeric_semantics.py -k quantize_linear`：RC `0`，`9 passed, 20 deselected in 1.07s`；
- `python -m pytest -q tests/test_operator_import_and_shape.py -k quantize`：RC `0`，`1 passed, 11 deselected in 0.92s`。

未运行 CUDA、numerical 或全量 pytest，不能把以上定向结果外推为全量门禁通过。
