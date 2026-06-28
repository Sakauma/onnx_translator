# /**
#   ******************************************************************************
#   * @file        release_check.py
#   * @author      Egor Izmaylov
#   * @brief       Validates release-readiness gates for packaging, ONNX coverage, and C backend coverage.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_ops import audit, strict_failures
from tools.abi_manifest import DEFAULT_MANIFEST, build_manifest, compare_manifests
from tools.benchmark_runtime import BENCHMARKS, _load_baseline
from tools.onnx_semantic_matrix import DEFAULT_JSON as ONNX_SEMANTIC_MATRIX_JSON
from tools.onnx_semantic_matrix import DEFAULT_MARKDOWN as ONNX_SEMANTIC_MATRIX_MARKDOWN
from tools.onnx_semantic_matrix import build_matrix as build_onnx_semantic_matrix
from tools.release_trend_history import validate_trend_history


REQUIRED_SCRIPTS = {
    "onnx-translator",
    "onnx-translator-abi-manifest",
    "onnx-translator-audit",
    "onnx-translator-verify",
    "onnx-translator-benchmark",
    "onnx-translator-model-smoke",
    "onnx-translator-release-check",
    "onnx-translator-package-smoke",
    "onnx-translator-release-artifacts",
    "onnx-translator-release-dashboard",
    "onnx-translator-release-preflight",
    "onnx-translator-release-trend-history",
    "onnx-translator-wheelhouse-smoke",
    "onnx-translator-onnx-semantic-matrix",
}

REQUIRED_MAKE_TARGETS = {
    "abi-check:",
    "benchmark:",
    "benchmark-baseline-check:",
    "benchmark-fixed-runner-check:",
    "benchmark-smoke:",
    "benchmark-smoke-report:",
    "manylinux-wheelhouse-check:",
    "manylinux-wheelhouse-check-full:",
    "manylinux-wheels:",
    "manylinux-wheels-full:",
    "model-smoke:",
    "onnx-semantic-matrix:",
    "package-smoke:",
    "release-artifacts:",
    "release-dashboard:",
    "release-preflight:",
    "release-trend-history:",
    "release-trend-history-refresh:",
    "sanitize:",
    "release-check:",
    "verify-cuda-smoke:",
    "verify-cuda-full:",
}

REQUIRED_FILES = [
    "constraints.txt",
    "docs/abi_manifest.json",
    "tools/abi_manifest.py",
    "tools/audit_operator_data.py",
    "tools/benchmark_runtime.py",
    "tools/model_suite.py",
    "tools/onnx_semantic_matrix.py",
    "tools/package_smoke.py",
    "tools/release_artifacts.py",
    "tools/release_dashboard.py",
    "tools/release_preflight.py",
    "tools/release_trend_history.py",
    "tools/run_sanitized_tests.py",
    "tools/wheelhouse_smoke.py",
    "tensor_ops/tensor_ops_activation_extra.c",
    "tensor_ops/tensor_ops_compare_logic.c",
    "tensor_ops/tensor_ops_conv_quant.c",
    "tensor_ops/tensor_ops_detection_sampling.c",
    "tensor_ops/tensor_ops_dtype.h",
    "tensor_ops/tensor_ops_global_pool.c",
    "tensor_ops/tensor_ops_layout_sequence.c",
    "tensor_ops/tensor_ops_loss.c",
    "tensor_ops/tensor_ops_pool_roi.c",
    "tensor_ops/tensor_ops_recurrent.c",
    "tensor_ops/tensor_ops_shape_grid.c",
    "tensor_ops/tensor_ops_softmax_family.c",
    "docs/performance_baseline.json",
    "docs/performance_fixed_runner_baseline.json",
    "docs/onnx_semantic_matrix.json",
    "docs/onnx_semantic_matrix.md",
    "docs/release_evidence_checklist.md",
    "docs/release_trend_history.json",
    "docs/release_trend_manifest.json",
    ".github/workflows/ci.yml",
    ".github/workflows/cuda.yml",
    ".github/workflows/performance.yml",
    ".github/workflows/wheels.yml",
    "docs/release.md",
    "requirements-dev.txt",
]

REQUIRED_MANYLINUX_PYTHON_TAGS = {"cp310", "cp311", "cp312"}
C_BACKEND_MAX_SHARD_LINES = 550
REQUIRED_TREND_WINDOW_IDS = {
    "cuda_full",
    "fixed_runner_performance",
    "manylinux_full",
    "release_evidence",
}
REQUIRED_FAILURE_TRIAGE_FIELDS = {
    "commit_sha",
    "failed_gate",
    "follow_up",
    "owner",
    "resolution",
    "root_cause",
    "run_url",
}


def _stable_semantic_matrix(payload: dict[str, object]) -> dict[str, object]:
    stable = dict(payload)
    stable.pop("generated_at", None)
    return stable


def _check_onnx_semantic_matrix() -> tuple[dict[str, object], list[str]]:
    failures = []
    current_payload, matrix_failures = build_onnx_semantic_matrix()
    if matrix_failures:
        failures.append("ONNX semantic matrix has weak rows: " + ", ".join(matrix_failures))
    if current_payload["verified_count"] != current_payload["official_onnx_latest_count"]:
        failures.append("ONNX semantic matrix verified_count must equal official latest count")
    if current_payload["missing_or_weak_count"] != 0:
        failures.append("ONNX semantic matrix must not contain runtime-only or weak rows")
    if current_payload["deprecated_alias_count"] < 2:
        failures.append("ONNX semantic matrix must record deprecated aliases for Scatter and Upsample")

    if not ONNX_SEMANTIC_MATRIX_JSON.exists():
        failures.append("ONNX semantic matrix JSON report is missing")
    else:
        try:
            stored_payload = json.loads(ONNX_SEMANTIC_MATRIX_JSON.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"ONNX semantic matrix JSON report is not readable: {exc}")
        else:
            if _stable_semantic_matrix(stored_payload) != _stable_semantic_matrix(current_payload):
                failures.append("ONNX semantic matrix JSON report is stale; rerun make onnx-semantic-matrix")

    if not ONNX_SEMANTIC_MATRIX_MARKDOWN.exists():
        failures.append("ONNX semantic matrix Markdown report is missing")
    else:
        text = ONNX_SEMANTIC_MATRIX_MARKDOWN.read_text(encoding="utf-8")
        for required in ["Deprecated aliases covered by canonical classes", "Scatter", "Upsample"]:
            if required not in text:
                failures.append(f"ONNX semantic matrix Markdown report is missing {required!r}")

    return current_payload, failures


def _load_pyproject() -> dict:
    path = ROOT / "pyproject.toml"
    if not path.exists():
        raise FileNotFoundError("pyproject.toml is required for release packaging metadata")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _check_pyproject(pyproject: dict) -> list[str]:
    failures = []
    project = pyproject.get("project", {})
    if project.get("name") != "onnx-translator":
        failures.append("project.name must be onnx-translator")
    if not project.get("version"):
        failures.append("project.version is required")
    if not str(project.get("requires-python", "")).startswith(">="):
        failures.append("project.requires-python must declare a lower bound")
    dependencies = set(project.get("dependencies", []))
    for dependency in ["numpy", "onnx", "onnxscript", "torch", "ml_dtypes"]:
        if not any(item.startswith(dependency) for item in dependencies):
            failures.append(f"project.dependencies is missing {dependency}")
    scripts = set(project.get("scripts", {}))
    missing_scripts = sorted(REQUIRED_SCRIPTS - scripts)
    if missing_scripts:
        failures.append("project.scripts is missing: " + ", ".join(missing_scripts))
    return failures


def _check_cibuildwheel_config(pyproject: dict, requirements_dev_text: str) -> list[str]:
    failures = []
    cibuildwheel = pyproject.get("tool", {}).get("cibuildwheel")
    if not cibuildwheel:
        return ["pyproject.toml must define cibuildwheel manylinux settings"]

    build_tags = set(str(cibuildwheel.get("build", "")).split())
    required_build_tags = {f"{tag}-manylinux_x86_64" for tag in REQUIRED_MANYLINUX_PYTHON_TAGS}
    missing_build_tags = sorted(required_build_tags - build_tags)
    if missing_build_tags:
        failures.append("cibuildwheel.build is missing: " + ", ".join(missing_build_tags))
    if "musllinux" not in str(cibuildwheel.get("skip", "")):
        failures.append("cibuildwheel.skip must exclude musllinux wheels until they are explicitly supported")
    if cibuildwheel.get("test-command"):
        failures.append("cibuildwheel.test-command must stay unset; use wheelhouse-smoke after build instead")
    before_build = str(cibuildwheel.get("before-build", ""))
    for generated_path in ["build", "onnx_translator.egg-info", "tensor_ops.so", "tensor_ops_asan.so"]:
        if generated_path not in before_build:
            failures.append(f"cibuildwheel.before-build must clean {generated_path}")

    linux_config = cibuildwheel.get("linux", {})
    if "x86_64" not in linux_config.get("archs", []):
        failures.append("cibuildwheel.linux.archs must include x86_64")
    if linux_config.get("manylinux-x86_64-image") != "manylinux2014":
        failures.append("cibuildwheel.linux.manylinux-x86_64-image must be manylinux2014")
    if "cibuildwheel" not in requirements_dev_text:
        failures.append("requirements-dev.txt must include cibuildwheel")
    return failures


def _check_release_evidence_checklist() -> list[str]:
    path = ROOT / "docs" / "release_evidence_checklist.md"
    if not path.exists():
        return ["release evidence checklist is missing"]
    text = path.read_text(encoding="utf-8")
    required_tokens = [
        "Release Evidence Checklist",
        "release candidate commit SHA",
        "result/release_preflight.json",
        "result/release_preflight_plan.json",
        "result/release_dashboard.md",
        "docs/release_trend_manifest.json",
        "docs/release_trend_history.json",
        "release-evidence",
        "manylinux-wheels-full",
        "benchmark-fixed-runner-check",
        "verify-cuda-full",
        "sanitize",
        "CI run",
    ]
    return [f"release evidence checklist is missing {token!r}" for token in required_tokens if token not in text]


def _check_release_evidence_workflow() -> list[str]:
    path = ROOT / ".github" / "workflows" / "ci.yml"
    if not path.exists():
        return ["release evidence workflow is missing .github/workflows/ci.yml"]
    text = path.read_text(encoding="utf-8")
    required_tokens = [
        "Build release evidence dashboard",
        "Upload release evidence dashboard",
        "workflow_dispatch",
        "release-evidence-${{ github.run_id }}",
        "result/release_preflight_plan.json",
        "result/release_dashboard.md",
        "result/release_dashboard.json",
        "docs/release_evidence_checklist.md",
        "docs/release_trend_history.json",
        "docs/release_trend_manifest.json",
        "retention-days: 90",
        "--include-cuda-smoke",
        "--include-cuda-full",
        "--include-manylinux",
        "--include-manylinux-full",
        "--include-fixed-runner-perf",
    ]
    return [f"release evidence workflow is missing {token!r}" for token in required_tokens if token not in text]


def _check_release_trend_manifest() -> list[str]:
    path = ROOT / "docs" / "release_trend_manifest.json"
    if not path.exists():
        return ["release trend manifest is missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"release trend manifest is not readable: {exc}"]

    failures = []
    if payload.get("schema_version") != 1:
        failures.append("release trend manifest must use schema_version=1")
    if int(payload.get("minimum_history_for_top_tier_release", 0)) < 3:
        failures.append("release trend manifest must require at least 3 historical runs for top-tier release")

    triage_fields = set(payload.get("failure_triage_fields", []))
    missing_triage = sorted(REQUIRED_FAILURE_TRIAGE_FIELDS - triage_fields)
    if missing_triage:
        failures.append("release trend manifest is missing failure triage fields: " + ", ".join(missing_triage))

    windows = payload.get("trend_windows", [])
    if not isinstance(windows, list):
        return failures + ["release trend manifest trend_windows must be a list"]

    windows_by_id = {str(window.get("id")): window for window in windows if isinstance(window, dict)}
    missing_windows = sorted(REQUIRED_TREND_WINDOW_IDS - set(windows_by_id))
    if missing_windows:
        failures.append("release trend manifest is missing trend windows: " + ", ".join(missing_windows))

    for window_id, window in sorted(windows_by_id.items()):
        workflow = str(window.get("source_workflow", ""))
        artifact_pattern = str(window.get("artifact_pattern", ""))
        retention_days = int(window.get("retention_days", 0))
        required_payloads = [str(item) for item in window.get("required_payloads", [])]
        workflow_tokens = [str(item) for item in window.get("workflow_tokens", [])]

        if not workflow:
            failures.append(f"release trend window {window_id} must declare source_workflow")
            continue
        workflow_path = ROOT / workflow
        if not workflow_path.exists():
            failures.append(f"release trend window {window_id} references missing workflow {workflow}")
            continue
        if not artifact_pattern:
            failures.append(f"release trend window {window_id} must declare artifact_pattern")
        if retention_days < 90:
            failures.append(f"release trend window {window_id} must retain artifacts for at least 90 days")
        if window_id == "fixed_runner_performance" and retention_days < 180:
            failures.append("fixed runner performance trend must retain artifacts for at least 180 days")
        if not required_payloads:
            failures.append(f"release trend window {window_id} must declare required_payloads")

        workflow_text = workflow_path.read_text(encoding="utf-8")
        required_tokens = [artifact_pattern, f"retention-days: {retention_days}", *required_payloads, *workflow_tokens]
        for token in sorted({token for token in required_tokens if token}):
            if token not in workflow_text:
                failures.append(f"release trend window {window_id} workflow {workflow} is missing {token!r}")
    return failures


def _check_release_trend_history() -> list[str]:
    _, failures = validate_trend_history()
    return failures


def _c_backend_shard_lines() -> list[dict[str, object]]:
    shards = []
    for path in sorted((ROOT / "tensor_ops").glob("tensor_ops_*.c")):
        shards.append(
            {
                "path": str(path.relative_to(ROOT)),
                "lines": len(path.read_text(encoding="utf-8").splitlines()),
            }
        )
    return shards


def _check_c_backend_shard_budgets() -> list[str]:
    shards = _c_backend_shard_lines()
    if not shards:
        return ["C backend shard budget cannot find tensor_ops/tensor_ops_*.c files"]
    failures = []
    for shard in shards:
        if int(shard["lines"]) > C_BACKEND_MAX_SHARD_LINES:
            failures.append(
                f"C backend shard {shard['path']} has {shard['lines']} lines; "
                f"limit is {C_BACKEND_MAX_SHARD_LINES}"
            )
    return failures


def _check_heavy_gate_artifact_retention() -> list[str]:
    makefile_text = (ROOT / "Makefile").read_text(encoding="utf-8")
    if "verify-cuda-smoke:\n\t$(PYTHON) tools/verify_all.py --iterations 3 --keep-artifacts" not in makefile_text:
        return ["verify-cuda-smoke must preserve cache artifacts for CUDA evidence upload"]
    if "verify-cuda-full:\n\t$(PYTHON) tools/verify_all.py --iterations 3 --keep-artifacts" not in makefile_text:
        return ["verify-cuda-full must preserve cache artifacts for CUDA evidence upload"]

    required_by_file = {
        ".github/workflows/cuda.yml": [
            "Upload CUDA smoke evidence",
            "Upload full CUDA evidence",
            "cuda-smoke-${{ github.run_id }}",
            "cuda-full-${{ github.run_id }}",
            "retention-days: 90",
        ],
        ".github/workflows/performance.yml": [
            "benchmark-fixed-runner-${{ github.run_id }}",
            "retention-days: 180",
        ],
        ".github/workflows/wheels.yml": [
            "manylinux-smoke-${{ github.run_id }}",
            "manylinux-${{ matrix.cibw-build }}-${{ github.run_id }}",
            "retention-days: 90",
        ],
    }
    failures = []
    for path_text, tokens in required_by_file.items():
        path = ROOT / path_text
        if not path.exists():
            failures.append(f"heavy gate workflow is missing {path_text}")
            continue
        text = path.read_text(encoding="utf-8")
        failures.extend(f"heavy gate workflow {path_text} is missing {token!r}" for token in tokens if token not in text)
    return failures


def build_release_summary() -> tuple[dict[str, object], list[str]]:
    pyproject = _load_pyproject()
    infos, metadata = audit()
    failures = _check_pyproject(pyproject)
    requirements_dev = ROOT / "requirements-dev.txt"
    requirements_dev_text = requirements_dev.read_text(encoding="utf-8") if requirements_dev.exists() else ""
    failures.extend(_check_cibuildwheel_config(pyproject, requirements_dev_text))
    failures.extend(_check_release_evidence_checklist())
    failures.extend(_check_release_evidence_workflow())
    failures.extend(_check_release_trend_manifest())
    trend_history, trend_history_failures = validate_trend_history()
    failures.extend(trend_history_failures)
    failures.extend(_check_heavy_gate_artifact_retention())
    failures.extend(_check_c_backend_shard_budgets())
    failures.extend(strict_failures(infos, metadata))
    semantic_matrix, semantic_failures = _check_onnx_semantic_matrix()
    failures.extend(semantic_failures)
    runtime_library = ROOT / "tensor_ops.so"
    setup_py = ROOT / "setup.py"
    manifest = ROOT / "MANIFEST.in"
    makefile_text = (ROOT / "Makefile").read_text(encoding="utf-8")
    setup_py_text = setup_py.read_text(encoding="utf-8") if setup_py.exists() else ""
    package_smoke_text = (ROOT / "tools" / "package_smoke.py").read_text(encoding="utf-8")
    release_artifacts_text = (ROOT / "tools" / "release_artifacts.py").read_text(encoding="utf-8")
    sanitizer_text = (ROOT / "tools" / "run_sanitized_tests.py").read_text(encoding="utf-8")

    if not runtime_library.exists():
        failures.append("tensor_ops.so must be built before release-check")
    if not setup_py.exists():
        failures.append("setup.py build hook is required to package tensor_ops.so")
    if not manifest.exists():
        failures.append("MANIFEST.in is required so source distributions include C/CUDA sources")
    for path in REQUIRED_FILES:
        if not (ROOT / path).exists():
            failures.append(f"release infrastructure file is missing: {path}")
    for target in sorted(REQUIRED_MAKE_TARGETS):
        if target not in makefile_text:
            failures.append(f"Makefile release target is missing: {target.rstrip(':')}")
    if "root_is_pure = False" not in setup_py_text:
        failures.append("runtime wheel must be marked platform-specific, not py3-none-any")
    if "has_ext_modules" not in setup_py_text or "BinaryDistribution" not in setup_py_text:
        failures.append("runtime wheel build must force a binary distribution so shared libraries land in platlib")
    if "_assert_runtime_wheel" not in package_smoke_text:
        failures.append("package smoke must reject platform-independent runtime wheels")
    if ".data/purelib/" not in package_smoke_text:
        failures.append("package smoke must reject shared libraries installed from purelib")
    if "_assert_runtime_wheel" not in release_artifacts_text:
        failures.append("release artifacts smoke must reject platform-independent runtime wheels")
    if "tools/model_suite.py" not in sanitizer_text:
        failures.append("sanitizer gate must run the representative model suite")
    performance_baseline = ROOT / "docs" / "performance_baseline.json"
    if performance_baseline.exists():
        try:
            baseline = _load_baseline(performance_baseline)
        except Exception as exc:
            failures.append(f"performance baseline is not readable: {exc}")
        else:
            missing_benchmarks = sorted(set(BENCHMARKS) - set(baseline))
            extra_benchmarks = sorted(set(baseline) - set(BENCHMARKS))
            if missing_benchmarks:
                failures.append("performance baseline is missing: " + ", ".join(missing_benchmarks))
            if extra_benchmarks:
                failures.append("performance baseline has unknown benchmarks: " + ", ".join(extra_benchmarks))
            for name, result in sorted(baseline.items()):
                if float(result.get("elements_per_second", 0.0)) <= 0.0:
                    failures.append(f"performance baseline for {name} must have positive throughput")
    fixed_runner_baseline = ROOT / "docs" / "performance_fixed_runner_baseline.json"
    if fixed_runner_baseline.exists():
        try:
            fixed_payload = json.loads(fixed_runner_baseline.read_text(encoding="utf-8"))
            fixed_baseline = _load_baseline(fixed_runner_baseline)
        except Exception as exc:
            failures.append(f"fixed-runner performance baseline is not readable: {exc}")
        else:
            if fixed_payload.get("baseline_kind") != "fixed_runner":
                failures.append("fixed-runner performance baseline must declare baseline_kind=fixed_runner")
            if not fixed_payload.get("runner_id"):
                failures.append("fixed-runner performance baseline must declare runner_id")
            missing_benchmarks = sorted(set(BENCHMARKS) - set(fixed_baseline))
            if missing_benchmarks:
                failures.append("fixed-runner performance baseline is missing: " + ", ".join(missing_benchmarks))
            for name, result in sorted(fixed_baseline.items()):
                if float(result.get("elements_per_second", 0.0)) <= 0.0:
                    failures.append(f"fixed-runner performance baseline for {name} must have positive throughput")
    if DEFAULT_MANIFEST.exists():
        expected_abi = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        failures.extend(compare_manifests(expected_abi, build_manifest()))

    if metadata["official_onnx_latest_supported_count"] != metadata["official_onnx_latest_count"]:
        failures.append("ONNX latest default-domain import coverage is incomplete")
    if metadata["active_python_only_runtime_count"] != 0:
        failures.append("ordinary tensor/numeric operators still have Python-only runtime paths")
    if metadata["cuda_verifier_count"] != metadata["numerical_plan_count"]:
        failures.append("CUDA verifier count does not match active numerical plan coverage")

    summary = {
        "project_name": pyproject["project"].get("name"),
        "project_version": pyproject["project"].get("version"),
        "operator_class_count": metadata["operator_class_count"],
        "onnx_latest_supported": metadata["official_onnx_latest_supported_count"],
        "onnx_latest_total": metadata["official_onnx_latest_count"],
        "c_runtime_count": metadata["c_runtime_count"],
        "python_only_active": metadata["active_python_only_runtime_count"],
        "cuda_verifier_count": metadata["cuda_verifier_count"],
        "numerical_plan_count": metadata["numerical_plan_count"],
        "numerical_plan_total_count": metadata["numerical_plan_total_count"],
        "mixed_precision_plan_count": metadata["mixed_precision_plan_count"],
        "onnx_semantic_verified": semantic_matrix["verified_count"],
        "onnx_semantic_missing_or_weak": semantic_matrix["missing_or_weak_count"],
        "onnx_semantic_deprecated_aliases": semantic_matrix["deprecated_alias_count"],
        "runtime_library": str(runtime_library),
        "runtime_library_present": runtime_library.exists(),
        "c_backend_shard_line_limit": C_BACKEND_MAX_SHARD_LINES,
        "c_backend_largest_shards": sorted(
            _c_backend_shard_lines(),
            key=lambda item: int(item["lines"]),
            reverse=True,
        )[:5],
        "trend_history": trend_history,
        "release_infrastructure_files": REQUIRED_FILES,
        "release_make_targets": sorted(target.rstrip(":") for target in REQUIRED_MAKE_TARGETS),
        "manylinux_python_tags": sorted(REQUIRED_MANYLINUX_PYTHON_TAGS),
        "cibuildwheel_manylinux_image": pyproject.get("tool", {})
        .get("cibuildwheel", {})
        .get("linux", {})
        .get("manylinux-x86_64-image"),
        "failures": failures,
    }
    return summary, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run release-readiness checks.")
    parser.add_argument("--json", dest="json_path", help="Optional path to write a release summary JSON file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary, failures = build_release_summary()

    print("Release readiness summary:")
    print(f"- project: {summary['project_name']} {summary['project_version']}")
    print(f"- ONNX latest default-domain coverage: {summary['onnx_latest_supported']}/{summary['onnx_latest_total']}")
    print(f"- C runtime operator classes: {summary['c_runtime_count']}")
    print(f"- active Python-only ordinary runtime paths: {summary['python_only_active']}")
    print(f"- CUDA verifier / active numerical coverage: {summary['cuda_verifier_count']}/{summary['numerical_plan_count']}")
    print(f"- default numerical plans: {summary['numerical_plan_total_count']}")
    print(f"- mixed precision numerical plans: {summary['mixed_precision_plan_count']}")
    print(
        "- ONNX semantic matrix verified / weak / deprecated aliases: "
        f"{summary['onnx_semantic_verified']}/"
        f"{summary['onnx_semantic_missing_or_weak']}/"
        f"{summary['onnx_semantic_deprecated_aliases']}"
    )
    print(f"- runtime library present: {summary['runtime_library_present']} ({summary['runtime_library']})")
    largest_shards = ", ".join(
        f"{item['path']}={item['lines']}" for item in summary["c_backend_largest_shards"]
    )
    print(f"- C backend shard line limit: {summary['c_backend_shard_line_limit']} ({largest_shards})")
    print(
        "- release trend history top-tier ready: "
        f"{summary['trend_history'].get('all_windows_top_tier_ready', False)}"
    )
    print(f"- release Make targets: {', '.join(summary['release_make_targets'])}")

    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote release summary: {output}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("Release readiness gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
