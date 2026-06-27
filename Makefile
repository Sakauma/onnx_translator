# /**
#   ******************************************************************************
#   * @file        Makefile
#   * @author      Egor Izmaylov
#   * @brief       定义 C 后端构建、清理、测试和验证命令。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

# Makefile
CC = gcc
PYTHON ?= python3
# -O3: 最高级优化
# -fopenmp: 开启多线程并行
# -fPIC: 位置无关代码
# -Wall: 显示所有警告
CFLAGS ?= -O3 -fPIC -Wall -Wextra -fopenmp
LDFLAGS = -shared -lm
SANITIZER_CFLAGS ?= -O1 -g -fPIC -Wall -Wextra -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer
SANITIZER_LDFLAGS ?= -shared -lm -fsanitize=address,undefined
CUDA_SMOKE_OPS ?= add matmul conv2d softmax quantize_linear dequantize_linear
CUDA_SMOKE_ARGS = $(foreach op,$(CUDA_SMOKE_OPS),--op $(op))
CIBW_BUILD ?= cp312-manylinux_x86_64
CIBW_BUILD_FULL ?= cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64
CIBW_ARCHS_LINUX ?= x86_64
PERF_BASELINE ?= docs/performance_fixed_runner_baseline.json
PERF_MAX_REGRESSION ?= 0.10
PERF_REPEAT ?= 20
PERF_RUNNER_ID ?= fixed-linux-x64-perf
PERF_WARMUP ?= 5

# 目标文件
TARGET = tensor_ops.so
SANITIZER_TARGET = tensor_ops_asan.so
# 源文件目录
SRC_DIR = tensor_ops
# 自动查找所有 .c 文件
SRCS = $(wildcard $(SRC_DIR)/*.c)

all: $(TARGET)

.PHONY: all clean check test audit abi-check benchmark benchmark-baseline-check benchmark-fixed-runner-check benchmark-smoke benchmark-smoke-report manylinux-wheelhouse-check manylinux-wheelhouse-check-full manylinux-wheels manylinux-wheels-full model-smoke onnx-semantic-matrix package-smoke release-artifacts release-preflight sanitize release-check verify verify-cpu verify-cuda-smoke

$(TARGET): $(SRCS) $(SRC_DIR)/tensor_ops.h $(SRC_DIR)/tensor_ops_internal.h
	@echo "Compiling C extension..."
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)
	@echo "Build successful: $(TARGET)"

clean:
	rm -f $(TARGET) $(SANITIZER_TARGET) nn/$(TARGET)
	@echo "Cleaned up."

check: all
	$(PYTHON) -m py_compile \
		nn/__init__.py nn/Operators.py nn/ONNXImport.py nn/ModelInitParas.py nn/GraphVisualization.py \
		tools/__init__.py tools/cli.py tools/health_check.py tools/verify_all.py tools/audit_ops.py \
		tools/commands/__init__.py tools/commands/create_model.py tools/commands/create_graph_ops_model.py \
		tools/commands/graph_logic.py tools/commands/verify_graph.py tools/commands/numerical_correctness.py \
		tools/abi_manifest.py tools/benchmark_runtime.py tools/model_suite.py tools/onnx_semantic_matrix.py tools/package_smoke.py tools/release_artifacts.py tools/release_check.py tools/release_preflight.py tools/run_sanitized_tests.py tools/wheelhouse_smoke.py \
		tools/numerical/__init__.py tools/numerical/cli.py tools/numerical/compare.py tools/numerical/cuda.py \
		tools/numerical/data.py tools/numerical/dtype.py tools/numerical/runner.py
	@echo "Static Python compile check passed."

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) tools/audit_ops.py --output docs/reports/operator_coverage.md

abi-check:
	$(PYTHON) tools/abi_manifest.py --check

benchmark: all
	$(PYTHON) tools/benchmark_runtime.py

benchmark-baseline-check: all
	$(PYTHON) tools/benchmark_runtime.py --warmup 1 --repeat 3 --baseline docs/performance_baseline.json --max-regression 0 --json result/benchmark_baseline_check.json

benchmark-fixed-runner-check: all
	PERF_RUNNER_ID="$(PERF_RUNNER_ID)" $(PYTHON) tools/benchmark_runtime.py --warmup $(PERF_WARMUP) --repeat $(PERF_REPEAT) --baseline $(PERF_BASELINE) --baseline-kind fixed_runner --require-runner-id "$(PERF_RUNNER_ID)" --require-baseline-kind fixed_runner --max-regression $(PERF_MAX_REGRESSION) --json result/benchmark_fixed_runner_check.json

benchmark-smoke: all
	$(PYTHON) tools/benchmark_runtime.py --smoke --warmup 1 --repeat 3

benchmark-smoke-report: all
	$(PYTHON) tools/benchmark_runtime.py --smoke --warmup 1 --repeat 3 --json result/benchmark_smoke.json

manylinux-wheels:
	CIBW_BUILD="$(CIBW_BUILD)" CIBW_ARCHS_LINUX="$(CIBW_ARCHS_LINUX)" $(PYTHON) -m cibuildwheel --platform linux --output-dir wheelhouse

manylinux-wheelhouse-check:
	$(PYTHON) tools/wheelhouse_smoke.py wheelhouse --require-platform manylinux

manylinux-wheels-full:
	CIBW_BUILD="$(CIBW_BUILD_FULL)" CIBW_ARCHS_LINUX="$(CIBW_ARCHS_LINUX)" $(PYTHON) -m cibuildwheel --platform linux --output-dir wheelhouse

manylinux-wheelhouse-check-full:
	$(PYTHON) tools/wheelhouse_smoke.py wheelhouse --require-platform manylinux --require-python-tag cp310 --require-python-tag cp311 --require-python-tag cp312

model-smoke: all
	$(PYTHON) tools/model_suite.py

onnx-semantic-matrix:
	$(PYTHON) tools/onnx_semantic_matrix.py --check --json docs/onnx_semantic_matrix.json --markdown docs/onnx_semantic_matrix.md

package-smoke:
	$(PYTHON) tools/package_smoke.py

release-artifacts:
	$(PYTHON) tools/release_artifacts.py --keep-artifacts

release-preflight:
	$(PYTHON) tools/release_preflight.py

$(SANITIZER_TARGET): $(SRCS) $(SRC_DIR)/tensor_ops.h $(SRC_DIR)/tensor_ops_internal.h
	@echo "Compiling sanitized C backend..."
	$(CC) $(SANITIZER_CFLAGS) -o $@ $(SRCS) $(SANITIZER_LDFLAGS)
	@echo "Sanitized build successful: $(SANITIZER_TARGET)"

sanitize: $(SANITIZER_TARGET)
	$(PYTHON) tools/run_sanitized_tests.py

release-check: all abi-check
	$(PYTHON) tools/release_check.py

verify:
	$(PYTHON) tools/verify_all.py

verify-cpu:
	$(PYTHON) tools/verify_all.py --skip-cuda

verify-cuda-smoke:
	$(PYTHON) tools/verify_all.py --iterations 3 $(CUDA_SMOKE_ARGS)
