#!/usr/bin/env bash
# /**
#   ******************************************************************************
#   * @file        compile_cuda.sh
#   * @author      Egor Izmaylov
#   * @brief       查找 nvcc 并编译 cuda 目录下的所有 CUDA 验证程序，将可执行文件写入 cache 目录。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

set -euo pipefail

CUDA_DIR="${CUDA_DIR:-cuda}"
CACHE_DIR="${CACHE_DIR:-cache}"
NVCC="${NVCC:-nvcc}"
FORCE=0
OPS=()

usage() {
    cat <<'EOF'
Usage: compile_cuda.sh [--op OP]... [--force]

Compile CUDA verifier programs from cuda/ into cache/.

Options:
  --op OP    Compile only cuda/verify_<op>.cu. Can be repeated.
  --force    Recompile even when the cached executable is newer than its inputs.
  -h, --help Show this help text.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --op)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --op requires an operator name." >&2
                exit 2
            fi
            OPS+=("$2")
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# 定位 `resolve_nvcc` 所需工具或执行入口，让后续构建步骤在缺少依赖时给出明确诊断。
resolve_nvcc() {
    if command -v "${NVCC}" >/dev/null 2>&1; then
        command -v "${NVCC}"
        return 0
    fi

    shopt -s nullglob
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [ -x "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! NVCC_BIN="$(resolve_nvcc)"; then
    echo "ERROR: nvcc not found. Set NVCC=/path/to/nvcc, add it to PATH, or install CUDA under /usr/local/cuda*." >&2
    exit 127
fi

if [ ! -d "${CUDA_DIR}" ]; then
    echo "ERROR: CUDA source directory not found: ${CUDA_DIR}" >&2
    exit 1
fi

shopt -s nullglob
all_cuda_files=("${CUDA_DIR}"/*.cu)
cuda_files=()
if [ ${#OPS[@]} -eq 0 ]; then
    cuda_files=("${all_cuda_files[@]}")
else
    for op_name in "${OPS[@]}"; do
        normalized="${op_name#verify_}"
        normalized="${normalized%.cu}"
        normalized="${normalized,,}"
        file="${CUDA_DIR}/verify_${normalized}.cu"
        if [ ! -f "${file}" ]; then
            echo "ERROR: CUDA verifier not found for --op ${op_name}: ${file}" >&2
            exit 1
        fi
        cuda_files+=("${file}")
    done
fi
if [ ${#cuda_files[@]} -eq 0 ]; then
    echo "ERROR: no CUDA source files selected in ${CUDA_DIR}" >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}"

common_headers=("${CUDA_DIR}"/*.cuh)

is_fresh() {
    local output="$1"
    shift
    if [ "${FORCE}" -eq 1 ] || [ ! -x "${output}" ]; then
        return 1
    fi
    local dep
    for dep in "$@"; do
        if [ "${dep}" -nt "${output}" ]; then
            return 1
        fi
    done
    return 0
}

echo "Compiling ${#cuda_files[@]} CUDA verifier(s) into ${CACHE_DIR}..."
failures=()
compiled=0
skipped=0
for file in "${cuda_files[@]}"; do
    filename="$(basename "${file}" .cu)"
    output="${CACHE_DIR}/${filename}"
    deps=("${file}" "$0" "${common_headers[@]}")
    if is_fresh "${output}" "${deps[@]}"; then
        echo "  ${file} -> ${output} (fresh, skipped)"
        skipped=$((skipped + 1))
        continue
    fi
    echo "  ${file} -> ${output}"
    if ! "${NVCC_BIN}" "${file}" -o "${output}"; then
        failures+=("${file}")
    else
        compiled=$((compiled + 1))
    fi
done

if [ ${#failures[@]} -ne 0 ]; then
    echo "ERROR: ${#failures[@]} CUDA verifier(s) failed to compile:" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "CUDA verifier compilation succeeded. compiled=${compiled} skipped=${skipped}"
