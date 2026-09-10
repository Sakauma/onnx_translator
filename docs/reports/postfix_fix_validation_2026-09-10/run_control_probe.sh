#!/usr/bin/env bash
set -o pipefail

report_dir="docs/reports/postfix_fix_validation_2026-09-10"
probe_dir="$report_dir/control_probe"
mkdir -p "$probe_dir"

printf '%s\n' \
    'command: run original import_graph/probe_control_sequence.py logic with OUT redirected to the new validation directory' \
    > "$report_dir/control_probe.command"

python -u -c '
import importlib.util
from pathlib import Path

source = Path("docs/reports/postfix_audit_2026-09-10/import_graph/probe_control_sequence.py")
spec = importlib.util.spec_from_file_location("probe_control_sequence", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = Path("docs/reports/postfix_fix_validation_2026-09-10/control_probe")
for case in (module.zero_length_scan, module.if_sequence_output, module.nested_ml_domain):
    try:
        case()
    except Exception as exc:
        print(case.__name__)
        print("case_setup_error", f"{type(exc).__name__}: {exc}")
' 2>&1 | tee "$report_dir/control_probe.stdout_stderr.log"
python_rc=${PIPESTATUS[0]}

# The original audit probe is a reporting probe and catches product exceptions.
# Promote any reported setup/product failure or missing actual line to a nonzero validation RC.
actual_count=$(grep -c '^actual ' "$report_dir/control_probe.stdout_stderr.log" || true)
semantic_rc=0
if (( python_rc != 0 || actual_count != 3 )); then
    semantic_rc=1
fi
if grep -Eq '^(actual|case_setup_error) .*Error' "$report_dir/control_probe.stdout_stderr.log"; then
    semantic_rc=1
fi
{
    printf 'python_rc=%s\n' "$python_rc"
    printf 'actual_case_count=%s\n' "$actual_count"
    printf 'semantic_rc=%s\n' "$semantic_rc"
} > "$report_dir/control_probe.rc"
exit "$semantic_rc"
