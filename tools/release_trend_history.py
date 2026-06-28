# /**
#   ******************************************************************************
#   * @file        release_trend_history.py
#   * @author      Egor Izmaylov
#   * @brief       Validates archived release trend run history snapshots.
#   * @details     2026.06.28  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "release_trend_manifest.json"
DEFAULT_HISTORY = ROOT / "docs" / "release_trend_history.json"

VALID_WINDOW_STATUSES = {"satisfied", "insufficient_history"}
REQUIRED_ARTIFACT_FIELDS = {
    "artifact_id",
    "digest",
    "expired",
    "expires_at",
    "name",
    "url",
}
REQUIRED_RUN_FIELDS = {
    "artifacts",
    "created_at",
    "event",
    "head_sha",
    "run_id",
    "run_url",
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_datetime(value: object, field_name: str, failures: list[str]) -> None:
    text = str(value)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        failures.append(f"{field_name} must be an ISO-8601 timestamp")


def _manifest_window_ids(manifest: dict[str, object]) -> set[str]:
    return {
        str(window.get("id"))
        for window in manifest.get("trend_windows", [])
        if isinstance(window, dict) and window.get("id")
    }


def _window_by_id(history: dict[str, object]) -> dict[str, dict[str, object]]:
    windows = {}
    for window in history.get("windows", []):
        if isinstance(window, dict) and window.get("id"):
            windows[str(window["id"])] = window
    return windows


def _validate_artifacts(window_id: str, run_id: object, artifacts: object, failures: list[str]) -> None:
    if not isinstance(artifacts, list) or not artifacts:
        failures.append(f"trend window {window_id} run {run_id} must list retained artifacts")
        return
    names = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            failures.append(f"trend window {window_id} run {run_id} has a non-object artifact")
            continue
        missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
        if missing:
            failures.append(
                f"trend window {window_id} run {run_id} artifact is missing: " + ", ".join(missing)
            )
        name = str(artifact.get("name", ""))
        if name in names:
            failures.append(f"trend window {window_id} run {run_id} repeats artifact {name}")
        names.add(name)
        if not str(artifact.get("url", "")).startswith("https://api.github.com/repos/"):
            failures.append(f"trend window {window_id} run {run_id} artifact {name} must use a GitHub API URL")
        if not str(artifact.get("digest", "")).startswith("sha256:"):
            failures.append(f"trend window {window_id} run {run_id} artifact {name} must record a sha256 digest")
        if artifact.get("expired") is not False:
            failures.append(f"trend window {window_id} run {run_id} artifact {name} must be retained and unexpired")
        if artifact.get("expires_at"):
            _parse_datetime(artifact["expires_at"], f"trend window {window_id} artifact {name} expires_at", failures)


def _validate_successful_runs(window_id: str, runs: object, failures: list[str]) -> None:
    if not isinstance(runs, list):
        failures.append(f"trend window {window_id} successful_runs must be a list")
        return

    seen_run_ids = set()
    for run in runs:
        if not isinstance(run, dict):
            failures.append(f"trend window {window_id} has a non-object successful run")
            continue
        missing = sorted(REQUIRED_RUN_FIELDS - set(run))
        if missing:
            failures.append(f"trend window {window_id} run is missing: " + ", ".join(missing))
        run_id = run.get("run_id")
        if run_id in seen_run_ids:
            failures.append(f"trend window {window_id} repeats run_id {run_id}")
        seen_run_ids.add(run_id)
        if not str(run.get("run_url", "")).startswith("https://github.com/"):
            failures.append(f"trend window {window_id} run {run_id} must use a GitHub run URL")
        if len(str(run.get("head_sha", ""))) != 40:
            failures.append(f"trend window {window_id} run {run_id} must record a full 40-character head_sha")
        if run.get("conclusion") != "success":
            failures.append(f"trend window {window_id} run {run_id} must have conclusion=success")
        if run.get("created_at"):
            _parse_datetime(run["created_at"], f"trend window {window_id} run {run_id} created_at", failures)
        _validate_artifacts(window_id, run_id, run.get("artifacts"), failures)


def validate_trend_history(
    history_path: Path = DEFAULT_HISTORY,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    if not manifest_path.exists():
        return {}, [f"release trend manifest is missing: {manifest_path}"]
    if not history_path.exists():
        return {}, [f"release trend history is missing: {history_path}"]

    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        return {}, [f"release trend manifest is not readable: {exc}"]
    try:
        history = _read_json(history_path)
    except Exception as exc:
        return {}, [f"release trend history is not readable: {exc}"]

    if history.get("schema_version") != 1:
        failures.append("release trend history must use schema_version=1")
    minimum_history = int(manifest.get("minimum_history_for_top_tier_release", 0))
    if int(history.get("minimum_history_for_top_tier_release", 0)) != minimum_history:
        failures.append("release trend history minimum history must match the trend manifest")
    if history.get("captured_at"):
        _parse_datetime(history["captured_at"], "release trend history captured_at", failures)
    if not history.get("repository"):
        failures.append("release trend history must declare repository")

    manifest_ids = _manifest_window_ids(manifest)
    windows = _window_by_id(history)
    missing_windows = sorted(manifest_ids - set(windows))
    extra_windows = sorted(set(windows) - manifest_ids)
    if missing_windows:
        failures.append("release trend history is missing windows: " + ", ".join(missing_windows))
    if extra_windows:
        failures.append("release trend history has unknown windows: " + ", ".join(extra_windows))

    all_ready = True
    window_summaries = []
    for window_id in sorted(manifest_ids & set(windows)):
        window = windows[window_id]
        status = str(window.get("status", ""))
        if status not in VALID_WINDOW_STATUSES:
            failures.append(f"trend window {window_id} has invalid status {status!r}")

        successful_runs = window.get("successful_runs", [])
        _validate_successful_runs(window_id, successful_runs, failures)
        success_count = len(successful_runs) if isinstance(successful_runs, list) else 0
        if int(window.get("successful_run_count", -1)) != success_count:
            failures.append(f"trend window {window_id} successful_run_count must match successful_runs length")
        if int(window.get("minimum_required", 0)) != minimum_history:
            failures.append(f"trend window {window_id} minimum_required must match the trend manifest")

        expected_status = "satisfied" if success_count >= minimum_history else "insufficient_history"
        if status and status != expected_status:
            failures.append(f"trend window {window_id} status must be {expected_status} for {success_count} runs")
        expected_ready = success_count >= minimum_history
        if bool(window.get("top_tier_ready", False)) != expected_ready:
            failures.append(f"trend window {window_id} top_tier_ready must match historical run sufficiency")
        if not expected_ready:
            all_ready = False
            if not window.get("next_action"):
                failures.append(f"trend window {window_id} must declare next_action while history is insufficient")

        latest = successful_runs[0] if isinstance(successful_runs, list) and successful_runs else {}
        window_summaries.append(
            {
                "id": window_id,
                "status": expected_status,
                "successful_run_count": success_count,
                "minimum_required": minimum_history,
                "latest_run_url": latest.get("run_url") if isinstance(latest, dict) else None,
                "next_action": window.get("next_action"),
            }
        )

    if bool(history.get("all_windows_top_tier_ready", False)) != all_ready:
        failures.append("release trend history all_windows_top_tier_ready must match window sufficiency")

    summary = {
        "history_path": str(history_path),
        "manifest_path": str(manifest_path),
        "captured_at": history.get("captured_at"),
        "repository": history.get("repository"),
        "minimum_history_for_top_tier_release": minimum_history,
        "all_windows_top_tier_ready": all_ready,
        "windows": window_summaries,
    }
    return summary, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate archived release trend history evidence.")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY), help="Path to release trend history JSON.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to release trend manifest JSON.")
    parser.add_argument("--json", dest="json_path", help="Optional path to write the validated summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary, failures = validate_trend_history(Path(args.history), Path(args.manifest))

    print("Release trend history summary:")
    print(f"- repository: {summary.get('repository', '-')}")
    print(f"- captured at: {summary.get('captured_at', '-')}")
    print(f"- minimum history: {summary.get('minimum_history_for_top_tier_release', '-')}")
    print(f"- all windows top-tier ready: {summary.get('all_windows_top_tier_ready', False)}")
    for window in summary.get("windows", []):
        print(
            "- {id}: {status} ({successful_run_count}/{minimum_required})".format(
                id=window["id"],
                status=window["status"],
                successful_run_count=window["successful_run_count"],
                minimum_required=window["minimum_required"],
            )
        )

    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote release trend history summary: {output}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("Release trend history check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
