#!/usr/bin/env bash
set -euo pipefail

CUDA_DIR="${CUDA_DIR:-cuda}"
CACHE_DIR="${CACHE_DIR:-cache}"
NVCC="${NVCC:-nvcc}"

# Egor Izmaylov: Function `resolve_nvcc` locates the CUDA compiler from NVCC, PATH, or common install roots, so verifier compilation fails with a clear diagnostic when nvcc is unavailable.
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
