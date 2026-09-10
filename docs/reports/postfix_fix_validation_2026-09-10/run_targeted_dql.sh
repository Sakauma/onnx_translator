#!/usr/bin/env bash
set -o pipefail
report_dir="docs/reports/postfix_fix_validation_2026-09-10"
printf '%s\n' 'command: python -u tools/cli.py numerical --op dynamic_quantize_linear --iterations 1 --skip-plots' > "$report_dir/dql_numerical.command"
python -u tools/cli.py numerical --op dynamic_quantize_linear --iterations 1 --skip-plots 2>&1 | tee "$report_dir/dql_numerical.stdout_stderr.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$report_dir/dql_numerical.rc"
exit "$rc"
