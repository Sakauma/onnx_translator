#!/usr/bin/env bash
# 文件功能：查找 nvcc 并编译 cuda 目录下的所有 CUDA 验证程序，将可执行文件写入 cache 目录。
# 作者：Egor Izmaylov
# 时间：2026-06-02
set -euo pipefail

CUDA_DIR="${CUDA_DIR:-cuda}"
CACHE_DIR="${CACHE_DIR:-cache}"
NVCC="${NVCC:-nvcc}"

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
cuda_files=("${CUDA_DIR}"/*.cu)
if [ ${#cuda_files[@]} -eq 0 ]; then
    echo "ERROR: no CUDA source files found in ${CUDA_DIR}" >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}"

echo "Compiling ${#cuda_files[@]} CUDA verifier(s) into ${CACHE_DIR}..."
failures=()
for file in "${cuda_files[@]}"; do
    filename="$(basename "${file}" .cu)"
    output="${CACHE_DIR}/${filename}"
    echo "  ${file} -> ${output}"
    if ! "${NVCC_BIN}" "${file}" -o "${output}"; then
        failures+=("${file}")
    fi
done

if [ ${#failures[@]} -ne 0 ]; then
    echo "ERROR: ${#failures[@]} CUDA verifier(s) failed to compile:" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "CUDA verifier compilation succeeded."
