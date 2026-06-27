# /**
#   ******************************************************************************
#   * @file        benchmark_runtime.py
#   * @author      Egor Izmaylov
#   * @brief       Runs repeatable C backend performance smoke benchmarks.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nn
from nn import Tensor
from nn.Operators import ADD, Conv, MatMul, ReduceSum, Softmax


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    elements: int
    factory: Callable[[np.random.Generator], Callable[[], object]]


def _result_elements(result: object, fallback: int) -> int:
    tensor = result.get("tensor") if isinstance(result, dict) else result
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    data = getattr(tensor, "data", None)
    return int(getattr(data, "size", fallback))


def _add_case(rng: np.random.Generator) -> Callable[[], object]:
    shape = (1024, 1024)
    a = Tensor(*shape, dtype="float32", data=rng.standard_normal(shape, dtype=np.float32))
    b = Tensor(*shape, dtype="float32", data=rng.standard_normal(shape, dtype=np.float32))
    op = ADD(["a", "b"], ["y"], dtype="float32")
    return lambda: op.forward(a, b)


def _matmul_case(rng: np.random.Generator) -> Callable[[], object]:
    a_shape = (256, 256)
    b_shape = (256, 256)
    a = Tensor(*a_shape, dtype="float32", data=rng.standard_normal(a_shape, dtype=np.float32))
    b = Tensor(*b_shape, dtype="float32", data=rng.standard_normal(b_shape, dtype=np.float32))
    op = MatMul(["a", "b"], ["y"], dtype="float32")
    return lambda: op.forward(a, b)


def _conv2d_case(rng: np.random.Generator) -> Callable[[], object]:
    x_shape = (1, 8, 32, 32)
    w_shape = (16, 8, 3, 3)
    x = Tensor(*x_shape, dtype="float32", data=rng.standard_normal(x_shape, dtype=np.float32))
    w = Tensor(*w_shape, dtype="float32", data=rng.standard_normal(w_shape, dtype=np.float32))
    b = Tensor(16, dtype="float32", data=rng.standard_normal((16,), dtype=np.float32))
    op = Conv(
        ["x", "w", "b"],
        ["y"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 1],
        group=1,
        dtype="float32",
    )
    return lambda: op.forward(x, w, b)


def _reduce_sum_case(rng: np.random.Generator) -> Callable[[], object]:
    shape = (64, 128, 16)
    x = Tensor(*shape, dtype="float32", data=rng.standard_normal(shape, dtype=np.float32))
    op = ReduceSum(["x"], ["y"], axes=[1], keepdims=0, dtype="float32")
    return lambda: op.forward(x)


def _softmax_case(rng: np.random.Generator) -> Callable[[], object]:
    shape = (512, 256)
    x = Tensor(*shape, dtype="float32", data=rng.standard_normal(shape, dtype=np.float32))
    op = Softmax(["x"], ["y"], axis=-1, dtype="float32")
    return lambda: op.forward(x)


BENCHMARKS = {
    "add": BenchmarkCase("add", 1024 * 1024, _add_case),
    "matmul": BenchmarkCase("matmul", 256 * 256, _matmul_case),
    "conv2d": BenchmarkCase("conv2d", 1 * 16 * 32 * 32, _conv2d_case),
    "reduce_sum": BenchmarkCase("reduce_sum", 64 * 16, _reduce_sum_case),
    "softmax": BenchmarkCase("softmax", 512 * 256, _softmax_case),
}

SMOKE_MIN_THROUGHPUT = {
    "add": 10_000_000.0,
    "matmul": 100_000.0,
    "conv2d": 100_000.0,
    "reduce_sum": 250_000.0,
    "softmax": 2_000_000.0,
}


def _result_map(results: list[dict[str, float | str | int]]) -> dict[str, dict[str, float | str | int]]:
    return {str(result["name"]): result for result in results}


def _parse_min_throughput(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid throughput threshold {value!r}; expected op=elements_per_second")
        name, raw_threshold = value.split("=", 1)
        if name not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {name!r}")
        thresholds[name] = float(raw_threshold)
    return thresholds


def _effective_thresholds(smoke: bool, values: list[str]) -> dict[str, float]:
    thresholds = dict(SMOKE_MIN_THROUGHPUT) if smoke else {}
    thresholds.update(_parse_min_throughput(values))
    return thresholds


def _load_baseline_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"schema_version": 0, "benchmarks": payload}
    if isinstance(payload, dict) and isinstance(payload.get("benchmarks"), list):
        return payload
    raise ValueError(f"unsupported benchmark baseline format: {path}")


def _load_baseline(path: Path) -> dict[str, dict[str, float | str | int]]:
    payload = _load_baseline_payload(path)
    return _result_map(payload["benchmarks"])


def _runner_id(args: argparse.Namespace) -> str:
    return str(getattr(args, "runner_id", None) or os.environ.get("PERF_RUNNER_ID") or "unscoped")


def _build_payload(
    results: list[dict[str, float | str | int]],
    args: argparse.Namespace,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    runner_id = _runner_id(args)
    return {
        "schema_version": 1,
        "baseline_kind": getattr(args, "baseline_kind", "ad_hoc"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "architecture": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
            "python_compiler": platform.python_compiler(),
        },
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "repeat": args.repeat,
        "runner_id": runner_id,
        "runtime_library": nn.TENSOR_OPS_LIB_PATH,
        "smoke": bool(getattr(args, "smoke", False)),
        "thresholds": thresholds or {},
        "warmup": args.warmup,
        "seed": args.seed,
        "benchmarks": results,
    }


def run_case(case: BenchmarkCase, warmup: int, repeat: int, rng: np.random.Generator) -> dict[str, float | str | int]:
    callable_case = case.factory(rng)
    last_result = None
    for _ in range(warmup):
        last_result = callable_case()

    samples_ms = []
    output_elements = case.elements
    for _ in range(repeat):
        start_ns = time.perf_counter_ns()
        last_result = callable_case()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        samples_ms.append(elapsed_ms)
        output_elements = _result_elements(last_result, case.elements)

    median_ms = statistics.median(samples_ms)
    throughput = output_elements / (median_ms / 1000.0)
    return {
        "name": case.name,
        "repeat": repeat,
        "warmup": warmup,
        "output_elements": output_elements,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "elements_per_second": throughput,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C backend performance smoke benchmarks.")
    parser.add_argument("--op", action="append", choices=sorted(BENCHMARKS), help="Benchmark only this op. Can be repeated.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per benchmark.")
    parser.add_argument("--repeat", type=int, default=10, help="Measured runs per benchmark.")
    parser.add_argument("--seed", type=int, default=20260627, help="Random seed for benchmark inputs.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply conservative throughput floors suitable for CI smoke regression checks.",
    )
    parser.add_argument("--json", dest="json_path", help="Optional path to write machine-readable results.")
    parser.add_argument("--write-baseline", help="Write a benchmark baseline JSON payload for future regression checks.")
    parser.add_argument("--baseline", help="Compare against a benchmark baseline JSON payload.")
    parser.add_argument(
        "--baseline-kind",
        default="ad_hoc",
        help="Semantic kind for emitted results, for example portable_ci_floor or fixed_runner.",
    )
    parser.add_argument("--runner-id", help="Stable fixed runner identifier. Defaults to PERF_RUNNER_ID.")
    parser.add_argument(
        "--require-runner-id",
        help="Fail unless the current run and baseline payload both use this runner_id.",
    )
    parser.add_argument(
        "--require-baseline-kind",
        help="Fail unless the baseline payload declares this baseline_kind.",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.20,
        help="Allowed throughput regression ratio when --baseline is used. Default: 0.20.",
    )
    parser.add_argument(
        "--min-throughput",
        action="append",
        default=[],
        help="Fail if a benchmark is below a threshold, formatted as op=elements_per_second.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeat <= 0 or args.warmup < 0:
        raise ValueError("--repeat must be positive and --warmup must be non-negative")
    if not 0.0 <= args.max_regression < 1.0:
        raise ValueError("--max-regression must be in [0.0, 1.0)")
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        print(f"ERROR: C backend library not found: {nn.TENSOR_OPS_LIB_PATH}. Run `make` first.", file=sys.stderr)
        return 1

    thresholds = _effective_thresholds(args.smoke, args.min_throughput)
    selected = args.op or sorted(BENCHMARKS)
    rng = np.random.default_rng(args.seed)
    results = [run_case(BENCHMARKS[name], args.warmup, args.repeat, rng) for name in selected]

    print("| op | median ms | min ms | max ms | elements/s |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result['name']} | {result['median_ms']:.3f} | {result['min_ms']:.3f} | "
            f"{result['max_ms']:.3f} | {result['elements_per_second']:.0f} |"
        )

    failures = []
    for result in results:
        threshold = thresholds.get(str(result["name"]))
        if threshold is not None and float(result["elements_per_second"]) < threshold:
            failures.append(f"{result['name']} throughput {result['elements_per_second']:.0f} < {threshold:.0f}")

    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline_payload = _load_baseline_payload(baseline_path)
        baseline = _result_map(baseline_payload["benchmarks"])
        if args.require_runner_id:
            current_runner_id = _runner_id(args)
            baseline_runner_id = baseline_payload.get("runner_id")
            if current_runner_id != args.require_runner_id:
                failures.append(f"current runner_id {current_runner_id!r} does not match required {args.require_runner_id!r}")
            if baseline_runner_id != args.require_runner_id:
                failures.append(
                    f"baseline runner_id {baseline_runner_id!r} does not match required {args.require_runner_id!r}"
                )
        if args.require_baseline_kind and baseline_payload.get("baseline_kind") != args.require_baseline_kind:
            failures.append(
                f"baseline_kind {baseline_payload.get('baseline_kind')!r} does not match required "
                f"{args.require_baseline_kind!r}"
            )
        for result in results:
            name = str(result["name"])
            if name not in baseline:
                failures.append(f"{name} is missing from benchmark baseline {baseline_path}")
                continue
            baseline_throughput = float(baseline[name]["elements_per_second"])
            current_throughput = float(result["elements_per_second"])
            minimum_allowed = baseline_throughput * (1.0 - args.max_regression)
            if current_throughput < minimum_allowed:
                failures.append(
                    f"{name} throughput regression: {current_throughput:.0f} < "
                    f"{minimum_allowed:.0f} ({args.max_regression:.0%} below baseline {baseline_throughput:.0f})"
                )

    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_build_payload(results, args, thresholds), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote benchmark results: {output}")

    if args.write_baseline:
        output = Path(args.write_baseline)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_build_payload(results, args, thresholds), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote benchmark baseline: {output}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
