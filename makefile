# Makefile
CC = gcc
# -O3: 最高级优化
# -fopenmp: 开启多线程并行
# -fPIC: 位置无关代码
# -Wall: 显示所有警告
CFLAGS = -O3 -fPIC -Wall -fopenmp
LDFLAGS = -shared -lm

# 目标文件
TARGET = tensor_ops.so
# 源文件目录
SRC_DIR = tensor_ops
# 自动查找所有 .c 文件
SRCS = $(wildcard $(SRC_DIR)/*.c)

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling C extension..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build successful: $(TARGET)"

clean:
	rm -f $(TARGET)
	@echo "Cleaned up."