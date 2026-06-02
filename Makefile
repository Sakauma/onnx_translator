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

# 目标文件
TARGET = tensor_ops.so
# 源文件目录
SRC_DIR = tensor_ops
# 自动查找所有 .c 文件
SRCS = $(wildcard $(SRC_DIR)/*.c)

all: $(TARGET)

.PHONY: all clean check test audit verify verify-cpu

$(TARGET): $(SRCS) $(SRC_DIR)/tensor_ops.h $(SRC_DIR)/tensor_ops_internal.h
	@echo "Compiling C extension..."
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)
	@echo "Build successful: $(TARGET)"

clean:
	rm -f $(TARGET)
	@echo "Cleaned up."

check: all
	$(PYTHON) -m py_compile \
		nn/__init__.py nn/Operators.py nn/ONNXImport.py nn/ModelInitParas.py nn/GraphVisualization.py \
		tools/__init__.py tools/cli.py tools/health_check.py tools/verify_all.py tools/audit_ops.py \
		tools/commands/__init__.py tools/commands/create_model.py tools/commands/create_graph_ops_model.py \
		tools/commands/graph_logic.py tools/commands/verify_graph.py tools/commands/numerical_correctness.py \
		tools/numerical/__init__.py tools/numerical/cli.py tools/numerical/compare.py tools/numerical/cuda.py \
		tools/numerical/data.py tools/numerical/dtype.py tools/numerical/runner.py
	@echo "Static Python compile check passed."

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) tools/audit_ops.py --output docs/reports/operator_coverage.md

verify:
	$(PYTHON) tools/verify_all.py

verify-cpu:
	$(PYTHON) tools/verify_all.py --skip-cuda
