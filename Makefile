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
		create_model.py create_graph_ops_model.py graph_logic.py verify_graph.py numerical_correctness.py \
		nn/__init__.py nn/Operators.py nn/ONNXImport.py nn/ModelInitParas.py nn/GraphVisualization.py \
		tools/health_check.py tools/verify_all.py tools/audit_ops.py
	@echo "Static Python compile check passed."

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) tools/audit_ops.py --output docs/reports/operator_coverage.md

verify:
	$(PYTHON) tools/verify_all.py

verify-cpu:
	$(PYTHON) tools/verify_all.py --skip-cuda
