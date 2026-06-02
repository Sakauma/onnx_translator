"""文件功能：提供 numerical_correctness 命令行入口和默认算子验证计划。
作者：Egor Izmaylov
时间：2026-06-02
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import nn
from nn.Operators import (
    Gemm, MaxPool, AveragePool, LpPool, GlobalAveragePool, GlobalMaxPool, GlobalLpPool,
    ADD, SUB, MUL, DIV, MatMul, MatMulInteger, QLinearMatMul,
    ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd,
    RELU, ABS, Pow, SQRT, Conv, ConvTranspose, ConvInteger, QLinearConv, ScatterND, Clip,
    Equal, Greater, Less, GreaterOrEqual, LessOrEqual,
    Gather, GatherElements, GatherND, COS, LOG, EXP, SIGMOID, TANH,
    Sin, Floor, Atan, Sign, Tan, Neg, Mod, Max, Min, Not, And, Or, Xor, IsNaN,
    CumSum, Softmax, NonZero, TopK, ArgMin, ArgMax, Resize, RandomUniformLike, Einsum,
    QuantizeLinear, DequantizeLinear, MaxUnpool,
)

from . import cuda as cuda_backend
from .cuda import CUDA_VERIFY_DIR
from .runner import verify_op


def build_default_plans():
    return [
    # ---- 四则运算 ----
    (ADD, "add", [(64,64), (64,64)], ["float32", "float32"], "float32"),
    (SUB, "sub", [(64,64), (64,64)], ["float32", "float32"], "float32"),
    (MUL, "mul", [(64,64), (64,64)], ["float32", "float32"], "float32"),
    (DIV, "div", [(64,64), (64,64)], ["float32", "float32"], "float32"),

    # ---- 常见广播----
    (ADD, "add", [(10, 10, 10), (10, 1)], ["float32", "float32"], "float32"),
    (SUB, "sub", [(4, 1, 16), (16,)], ["float32", "float32"], "float32"),

    # ---- 激活 ----
    (RELU, "relu", [(128,128)], ["float32"], "float32"),
    (ABS, "abs", [(128,128)], ["float32"], "float32"),

    # ---- Conv ----
    (Conv, "conv2d",[(1, 1, 5, 5), (1, 1, 3, 3), (1,)],["float32", "float32", "float32"], "float32",{"pads":[0,0,0,0], "strides":[1,1], "dilations":[1,1], "group":1}),
    (ConvInteger, "conv_integer",[(1, 2, 5, 5), (2, 2, 3, 3), (1,), (2,)],["uint8", "int8", "uint8", "int8"], "int32",{"pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "group":1}),
    (QLinearConv, "qlinear_conv",[(1, 2, 5, 5), (1,), (1,), (2, 2, 3, 3), (2,), (2,), (1,), (1,)],["uint8", "float32", "uint8", "uint8", "float32", "uint8", "float32", "uint8"], "uint8",{"pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "group":1}),
    (ConvTranspose, "conv_transpose",[(1, 2, 4, 4), (2, 3, 3, 3), (3,)],["float32", "float32", "float32"], "float32",{"pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "group":1, "output_padding":[1,1]}),

    # ---- Softmax ----
    (Softmax, "softmax",[(4, 64)], ["float32"], "float32", {"axis":-1}),

    # ---- Gemm ----
    (Gemm, "gemm",[(16, 32), (32, 8), (8,)], ["float32", "float32", "float32"], "float32",{"alpha":1.0, "beta":1.0, "transA":0, "transB":0}),

    # ---- MaxPool ----
    (MaxPool, "max_pool",[(1, 2, 16, 16)], ["float32"], "float32",{"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),
    (AveragePool, "average_pool",[(1, 2, 7, 7)], ["float32"], "float32",{"kernel_shape":[3,3], "pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "count_include_pad":1}),
    (LpPool, "lp_pool",[(1, 2, 7, 7)], ["float32"], "float32",{"kernel_shape":[3,3], "pads":[1,1,1,1], "strides":[2,2], "dilations":[1,1], "p":2}),
    (GlobalAveragePool, "global_average_pool",[(1, 3, 5, 4)], ["float32"], "float32"),
    (GlobalMaxPool, "global_max_pool",[(1, 3, 5, 4)], ["float32"], "float32"),
    (GlobalLpPool, "global_lp_pool",[(1, 3, 5, 4)], ["float32"], "float32", {"p": 2}),
    (MaxUnpool, "max_unpool",[(1, 1, 2, 2), (1, 1, 2, 2)], ["float32", "int64"], "float32",{"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),

    (Equal,   "equal",   [(64,64), (64,64)], ["float32", "float32"], "bool"),
    (Greater, "greater", [(64,64), (64,64)], ["float32", "float32"], "bool"),
    (Less,    "less",    [(64,64), (64,64)], ["float32", "float32"], "bool"),

    (Clip, "clip",[(64,64), (1,), (1,)],["float32", "float32", "float32"],"float32"),

    (QuantizeLinear, "quantize_linear", [(64,64), (1,), (1,)], ["float32", "float32", "int8"], "int8"),
    (DequantizeLinear, "dequantize_linear", [(64,64), (1,), (1,)], ["int8", "float32", "int8"], "float32"),

    (SQRT, "sqrt", [(64, 64)], ["float32"], "float32"),

    (Pow, "pow", [(64,64), (64,64)], ["float32", "float32"], "float32"),

    (MatMul, "matmul",[(32, 64), (64,16)],["float32", "float32"],"float32"),
    (MatMulInteger, "matmul_integer", [(4, 6), (6, 5), (4,), (5,)], ["uint8", "int8", "uint8", "int8"], "int32"),
    (QLinearMatMul, "qlinear_matmul", [(4, 6), (4,), (4,), (6, 5), (5,), (5,), (1,), (1,)], ["uint8", "float32", "uint8", "uint8", "float32", "uint8", "float32", "uint8"], "uint8"),

    (ReduceMean, "reduce_mean",[(32, 64)],["float32"], "float32"),

    (Gather, "gather",[(32, 64), (8,)],["float32", "int64"],"float32",{"axis": 0}),

    (ScatterND, "scatternd",[(32, 64), (16, 2), (16,)],["float32", "int64", "float32"],"float32"), 

    (SIGMOID, "sigmoid", [(256,256)], ["float32"], "float32"),
    (COS, "cos", [(256,256)], ["float32"], "float32"),
    (Sin, "sin", [(256,256)], ["float32"], "float32"),
    (LOG, "log", [(256,256)], ["float32"], "float32"),
    (Floor, "floor", [(256,256)], ["float32"], "float32"),
    (EXP, "exp", [(256,256)], ["float32"], "float32"),
    (Atan, "atan", [(256,256)], ["float32"], "float32"),
    (Sign, "sign", [(256,256)], ["float32"], "float32"),
    (Tan, "tan", [(256,256)], ["float32"], "float32"),
    (TANH, "tanh", [(256,256)], ["float32"], "float32"),
    (Neg, "neg", [(256,256)], ["float32"], "float32"),
    (Mod, "mod", [(256,256), (256,256)], ["float32", "float32"], "float32"),
    (Max, "max", [(256,256), (256,256)], ["float32", "float32"], "float32"),
    (Min, "min", [(256,256), (256,256)], ["float32", "float32"], "float32"),
    (IsNaN, "isnan", [(256,256)], ["float32"], "bool"),

    # 归约（简化：2D 全归约）
    (ReduceSum, "reduce_sum", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceMax, "reduce_max", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceMin, "reduce_min", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),
    (ReduceProd, "reduce_prod", [(128,128)], ["float32"], "float32", {"axes":None, "keepdims":0}),

    # 逻辑（bool 输入/输出）
    (Not, "not", [(256,256)], ["bool"], "bool"),
    (And, "and", [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (Or,  "or",  [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (Xor, "xor", [(256,256), (256,256)], ["bool", "bool"], "bool"),
    (GreaterOrEqual, "greater_or_equal", [(256,256), (256,256)], ["float32", "float32"], "bool"),
    (LessOrEqual, "less_or_equal", [(256,256), (256,256)], ["float32", "float32"], "bool"),

    # 索引
    (GatherElements, "gather_elements", [(64,64), (64,64)], ["float32", "int64"], "float32", {"axis":1}),
    (GatherND, "gathernd", [(64,64), (256,2)], ["float32", "int64"], "float32"),

    # 扫描
    (CumSum, "cumsum", [(1024,)], ["float32"], "float32", {"exclusive":0, "reverse":0}),

    (NonZero, "nonzero", [(64,64)], ["float32"], "int64"),

    (ArgMin, "argmin", [(64,64)], ["float32"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),

    (ArgMax, "argmax", [(64,64)], ["float32"], "int64", {"axis": 1, "keepdims": 0, "select_last_index": 0}),

    # Resize: x, roi, scales, sizes
    (Resize, "resize", [(1,3,8,8), (0,), (0,), (4,)], ["float32", "float32", "float32", "int64"], "float32", {"mode": "nearest", "coord_mode": "asymmetric", "nearest_mode": "floor", "sizes_value": [1,3,16,16]}),

    # Einsum: 当前固定主路径 ij,jk->ik
    (Einsum, "einsum", [(16,32), (32,8)], ["float32", "float32"], "float32", {"equation": "ij,jk->ik"}),

    (TopK, "topk", [(32, 64), (1,)], ["float32", "int64"], "float32",{"axis": 1, "largest": 1, "sorted": 1, "k_value": 8}),

    (RandomUniformLike, "random_uniform_like", [(32, 32)], ["float32"], "float32", {"low": -1.0, "high": 1.0, "seed": 123}),
]



def main(argv=None):
    parser = argparse.ArgumentParser(description="Run C backend vs CUDA reference numerical checks.")
    parser.add_argument("--iterations", type=int, default=20, help="Iterations per test plan.")
    parser.add_argument("--op", action="append", help="Run only the named op. Can be repeated.")
    parser.add_argument("--cuda-dir", default=CUDA_VERIFY_DIR, help="Directory containing verify_* CUDA executables.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip matplotlib histogram generation.")
    args = parser.parse_args(argv)
    cuda_backend.CUDA_VERIFY_DIR = args.cuda_dir

    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        print(f"ERROR: C backend library not found: {nn.TENSOR_OPS_LIB_PATH}")
        print("Run `make` before numerical verification.")
        sys.exit(2)

    plans = build_default_plans()

    if args.op:
        selected_ops = set(args.op)
        plans = [plan for plan in plans if len(plan) >= 2 and plan[1] in selected_ops]
        missing_ops = selected_ops - {plan[1] for plan in plans}
        if missing_ops:
            print(f"ERROR: requested op(s) not found in test plans: {sorted(missing_ops)}")
            sys.exit(2)

    if not plans:
        print("ERROR: no numerical test plans selected.")
        sys.exit(2)

    print("🚀 开始数值验证 ...")
    ops_stats = {}
    failed_ops = []
    for plan in plans:
        if len(plan) == 5:
            op_cls, op_name, shapes, dtypes, out_dtype = plan
            init_args = {}
        elif len(plan) == 6:
            op_cls, op_name, shapes, dtypes, out_dtype, init_args = plan
        else:
            print(f"⚠️ 跳过格式错误的测试计划: {plan}")
            failed_ops.append("<malformed-plan>")
            continue
        abs_errs, rel_errs, ok = verify_op(
            op_cls,
            op_name,
            shapes,
            dtypes,
            out_dtype,
            init_args=init_args,
            iterations=args.iterations,
        )
        if not ok:
            failed_ops.append(op_name)
        # 按算子名称聚合数据
        if op_name not in ops_stats:
            ops_stats[op_name] = {'abs': [], 'rel': []}
        ops_stats[op_name]['abs'].extend(abs_errs)
        ops_stats[op_name]['rel'].extend(rel_errs)
    if not args.skip_plots:
        print("\n📊 正在按算子绘制误差分布直方图...")
        for op_name, stats in ops_stats.items():
            valid_abs = [x for x in stats['abs'] if np.isfinite(x) and x >= 0]
            valid_rel = [x for x in stats['rel'] if np.isfinite(x) and x >= 0]

            if len(valid_abs) == 0 or len(valid_rel) == 0:
                print(f"⚠️ [{op_name.upper()}] 没有可用的有限误差数据，跳过绘图")
                continue

            plt.figure(figsize=(14, 6))
            # --- 子图 1: 绝对误差分布 ---
            plt.subplot(1, 2, 1)
            plt.hist(valid_abs, bins=50, color='skyblue', edgecolor='black', log=True)
            plt.title(f'Operator [{op_name.upper()}] - Absolute Error Dist')
            plt.xlabel('Max Absolute Error')
            plt.ylabel('Count (Log Scale)')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            # 标注 99% 分位数 (P99)
            if len(valid_abs) > 0:
                p99_abs = np.percentile(valid_abs, 99)
                plt.axvline(p99_abs, color='red', linestyle='dashed', linewidth=1)
                plt.text(p99_abs, plt.ylim()[1]*0.9, f' P99: {p99_abs:.2e}', color='red')
            # --- 子图 2: 相对误差分布 ---
            plt.subplot(1, 2, 2)
            plt.hist(valid_rel, bins=50, color='salmon', edgecolor='black', log=True)
            plt.title(f'Operator [{op_name.upper()}] - Relative Error Dist')
            plt.xlabel('Max Relative Error')
            plt.ylabel('Count (Log Scale)')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            # 标注 99% 分位数 (P99)
            if len(valid_rel) > 0:
                p99_rel = np.percentile(valid_rel, 99)
                plt.axvline(p99_rel, color='red', linestyle='dashed', linewidth=1)
                plt.text(p99_rel, plt.ylim()[1]*0.9, f' P99: {p99_rel:.2e}', color='red')

            plt.tight_layout()
            plt.close()
    print("\n📈 详细统计报告 (99th Percentile Summary):")
    print(f"{'Operator':<10} | {'Abs (99%)':<12} | {'Rel (99%)':<12} | {'Samples':<8}")
    print("-" * 50)
    for op_name, stats in ops_stats.items():
        if len(stats['abs']) > 0:
            p99_abs = np.percentile(stats['abs'], 99)
            p99_rel = np.percentile(stats['rel'], 99)
            count = len(stats['abs'])
            print(f"{op_name.upper():<10} | {p99_abs:.2e}     | {p99_rel:.2e}     | {count:<8}")
    print("-" * 50)

    if failed_ops:
        print(f"ERROR: numerical verification failed for: {sorted(set(failed_ops))}")
        sys.exit(1)
