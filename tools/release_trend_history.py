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
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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
MANAGED_WINDOW_FIELDS = {
    "minimum_required",
    "next_action",
    "status",
    "successful_run_count",
    "successful_runs",
    "top_tier_ready",
}
DEFAULT_NEXT_ACTIONS = {
    "release_evidence": (
        "Continue retaining release-evidence artifacts on every PR, main push, "
        "and release-candidate workflow dispatch."
    ),
    "cuda_full": (
        "Run the CUDA workflow by weekly schedule or workflow_dispatch on the self-hosted CUDA runner "
        "until three successful cuda-full artifacts are retained."
    ),
    "fixed_runner_performance": (
        "Run the Performance workflow by weekly schedule or workflow_dispatch on the fixed performance runner "
        "until three benchmark-fixed-runner artifacts are retained."
    ),
    "manylinux_full": (
        "Run Wheels by weekly schedule or workflow_dispatch so the manylinux-full matrix produces cp310, cp311, "
        "and cp312 artifacts for three retained cycles."
    ),
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _workflow_filename(path_text: str) -> str:
    return Path(path_text).name


def _artifact_pattern_regex(pattern: str, run_id: object) -> re.Pattern[str]:
    regex = re.escape(pattern)
    regex = regex.replace(re.escape("${{ github.run_id }}"), re.escape(str(run_id)))
    regex = regex.replace(re.escape("${{ matrix.cibw-build }}"), r"(?P<matrix>[^/]+)")
    return re.compile(f"^{regex}$")


def _required_matrix_values(window: dict[str, object]) -> set[str]:
    values = set()
    for token in window.get("workflow_tokens", []):
        text = str(token)
        if re.fullmatch(r"cp\d+-manylinux_x86_64", text):
            values.add(text)
    return values


def _matching_artifacts(window: dict[str, object], run_id: object, artifacts: list[dict[str, object]]) -> list[dict[str, object]]:
    pattern = _artifact_pattern_regex(str(window.get("artifact_pattern", "")), run_id)
    required_matrix_values = _required_matrix_values(window)
    matched = []
    matrix_values = set()
    for artifact in artifacts:
        if artifact.get("expired") is True:
            continue
        match = pattern.match(str(artifact.get("name", "")))
        if not match:
            continue
        matched.append(artifact)
        matrix_value = match.groupdict().get("matrix")
        if matrix_value:
            matrix_values.add(matrix_value)
    if required_matrix_values and not required_matrix_values <= matrix_values:
        return []
    return sorted(matched, key=lambda item: str(item.get("name", "")))


def _artifact_record(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_id": artifact.get("id"),
        "name": artifact.get("name"),
        "url": artifact.get("url"),
        "digest": artifact.get("digest"),
        "expires_at": artifact.get("expires_at"),
        "expired": bool(artifact.get("expired", False)),
    }


def _run_record(run: dict[str, object], artifacts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run_id": run.get("id"),
        "run_url": run.get("html_url"),
        "event": run.get("event"),
        "branch": run.get("head_branch"),
        "head_sha": run.get("head_sha"),
        "created_at": run.get("created_at"),
        "conclusion": run.get("conclusion"),
        "artifacts": [_artifact_record(artifact) for artifact in artifacts],
    }


def _window_status(success_count: int, minimum_history: int) -> str:
    return "satisfied" if success_count >= minimum_history else "insufficient_history"


class GitHubActionsClient:
    def __init__(self, repository: str, token: str | None = None) -> None:
        self.repository = repository
        self.token = token

    def _request_json(self, path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(f"https://api.github.com/repos/{self.repository}/{path}{query}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API request failed for {path}: HTTP {exc.code} {message}") from exc

    def list_workflows(self) -> list[dict[str, object]]:
        payload = self._request_json("actions/workflows", {"per_page": 100})
        return [workflow for workflow in payload.get("workflows", []) if isinstance(workflow, dict)]

    def list_workflow_runs(self, workflow_id: object, max_runs: int) -> list[dict[str, object]]:
        payload = self._request_json(
            f"actions/workflows/{workflow_id}/runs",
            {"per_page": min(max_runs, 100)},
        )
        return [run for run in payload.get("workflow_runs", []) if isinstance(run, dict)]

    def list_run_artifacts(self, run_id: object) -> list[dict[str, object]]:
        payload = self._request_json(f"actions/runs/{run_id}/artifacts", {"per_page": 100})
        return [artifact for artifact in payload.get("artifacts", []) if isinstance(artifact, dict)]


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


def refresh_trend_history(
    manifest: dict[str, object],
    existing_history: dict[str, object],
    client: GitHubActionsClient,
    max_runs: int = 50,
    captured_at: str | None = None,
) -> dict[str, object]:
    minimum_history = int(manifest.get("minimum_history_for_top_tier_release", 0))
    workflows_by_path = {str(workflow.get("path")): workflow for workflow in client.list_workflows()}
    workflows_by_filename = {_workflow_filename(path): workflow for path, workflow in workflows_by_path.items()}
    existing_windows = _window_by_id(existing_history)
    refreshed_windows = []

    for manifest_window in manifest.get("trend_windows", []):
        if not isinstance(manifest_window, dict):
            continue
        window_id = str(manifest_window.get("id", ""))
        workflow_path = str(manifest_window.get("source_workflow", ""))
        workflow = workflows_by_path.get(workflow_path) or workflows_by_filename.get(_workflow_filename(workflow_path))
        if not workflow:
            raise RuntimeError(f"Could not find GitHub Actions workflow for {workflow_path}")

        successful_runs = []
        for run in client.list_workflow_runs(workflow.get("id"), max_runs):
            if run.get("status") != "completed" or run.get("conclusion") != "success":
                continue
            artifacts = client.list_run_artifacts(run.get("id"))
            matched_artifacts = _matching_artifacts(manifest_window, run.get("id"), artifacts)
            if not matched_artifacts:
                continue
            successful_runs.append(_run_record(run, matched_artifacts))
            if len(successful_runs) >= minimum_history:
                break

        success_count = len(successful_runs)
        status = _window_status(success_count, minimum_history)
        previous_window = existing_windows.get(window_id, {})
        refreshed_window = {
            "id": window_id,
            "status": status,
            "minimum_required": minimum_history,
            "successful_run_count": success_count,
            "top_tier_ready": success_count >= minimum_history,
            "successful_runs": successful_runs,
        }
        for key, value in previous_window.items():
            if key not in MANAGED_WINDOW_FIELDS and key != "id":
                refreshed_window[key] = value
        refreshed_window["next_action"] = previous_window.get("next_action") or DEFAULT_NEXT_ACTIONS.get(window_id, "")
        refreshed_windows.append(refreshed_window)

    all_ready = all(bool(window.get("top_tier_ready")) for window in refreshed_windows)
    return {
        "schema_version": 1,
        "captured_at": captured_at or _utc_now_text(),
        "repository": client.repository,
        "source": "GitHub Actions REST API",
        "minimum_history_for_top_tier_release": minimum_history,
        "all_windows_top_tier_ready": all_ready,
        "windows": refreshed_windows,
    }


def validate_trend_history_payload(
    history: dict[str, object],
    manifest: dict[str, object],
    history_path: Path = DEFAULT_HISTORY,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []

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

        expected_status = _window_status(success_count, minimum_history)
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


def validate_trend_history(
    history_path: Path = DEFAULT_HISTORY,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, object], list[str]]:
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

    return validate_trend_history_payload(history, manifest, history_path, manifest_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate archived release trend history evidence.")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY), help="Path to release trend history JSON.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to release trend manifest JSON.")
    parser.add_argument("--repository", help="GitHub repository in owner/name form; defaults to history.repository.")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Environment variable containing a GitHub token.")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum recent workflow runs to scan per trend window.")
    parser.add_argument("--refresh", action="store_true", help="Refresh history from the GitHub Actions API before validating.")
    parser.add_argument("--write", action="store_true", help="Write a refreshed history snapshot back to --history.")
    parser.add_argument("--json", dest="json_path", help="Optional path to write the validated summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    history_path = Path(args.history)
    manifest_path = Path(args.manifest)

    if args.refresh:
        manifest = _read_json(manifest_path)
        existing_history = _read_json(history_path) if history_path.exists() else {}
        repository = args.repository or str(existing_history.get("repository", ""))
        if not repository:
            print("ERROR: --repository is required when the history file does not declare repository")
            return 1
        token = os.environ.get(args.token_env)
        client = GitHubActionsClient(repository, token)
        try:
            refreshed_history = refresh_trend_history(manifest, existing_history, client, args.max_runs)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            if not token:
                print(f"ERROR: set {args.token_env} to avoid anonymous GitHub API rate limits")
            return 1
        summary, failures = validate_trend_history_payload(refreshed_history, manifest, history_path, manifest_path)
        if not failures and args.write:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(refreshed_history, indent=2, sort_keys=False) + "\n", encoding="utf-8")
            print(f"Wrote refreshed release trend history: {history_path}")
    else:
        summary, failures = validate_trend_history(history_path, manifest_path)

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
