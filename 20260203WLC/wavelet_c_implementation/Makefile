CC = gcc
CFLAGS = -Wall -O2 -std=c99 -lm
SRCDIR = .
OBJDIR = obj

# 源文件
SOURCES = wavelet_transform.c csv_processor.c interactive.c
OBJECTS = $(SOURCES:.c=.o)

# 目标程序
TARGETS = wavelet_test wavelet_interactive

.PHONY: all clean test

all: $(TARGETS)

# 基本测试程序
wavelet_test: main.o wavelet_transform.o csv_processor.o
	$(CC) $(CFLAGS) -o $@ $^

# 交互式程序
wavelet_interactive: interactive_main.o wavelet_transform.o csv_processor.o interactive.o
	$(CC) $(CFLAGS) -o $@ $^

# 编译规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 依赖关系
main.o: main.c wavelet_transform.h
wavelet_transform.o: wavelet_transform.c wavelet_transform.h
csv_processor.o: csv_processor.c wavelet_transform.h
interactive.o: interactive.c wavelet_transform.h
interactive_main.o: interactive_main.c wavelet_transform.h

# 测试
test: wavelet_test
	./wavelet_test

# 清理
clean:
	rm -f *.o $(TARGETS)
	rm -f *.csv
	rm -rf $(OBJDIR)

# 安装（可选）
install: all
	cp wavelet_test /usr/local/bin/
	cp wavelet_interactive /usr/local/bin/

# 帮助
help:
	@echo "可用目标:"
	@echo "  all             - 编译所有程序"
	@echo "  wavelet_test    - 编译基本测试程序"
	@echo "  wavelet_interactive - 编译交互式程序"
	@echo "  test            - 运行基本测试"
	@echo "  clean           - 清理编译文件"
	@echo "  install         - 安装到系统目录"
	@echo "  help            - 显示此帮助信息"