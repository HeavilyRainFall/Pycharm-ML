import sys
print("Python版本:", sys.version)
print("当前工作目录:", sys.path[0])

# 测试基本依赖
try:
    import PyQt5
    print("✓ PyQt5 可用")
except ImportError:
    print("✗ PyQt5 不可用")

try:
    import pandas
    print("✓ pandas 可用")
except ImportError:
    print("✗ pandas 不可用")

try:
    import numpy
    print("✓ numpy 可用")
except ImportError:
    print("✗ numpy 不可用")

try:
    import matplotlib
    print("✓ matplotlib 可用")
except ImportError:
    print("✗ matplotlib 不可用")

try:
    import pywt
    print("✓ pywt 可用")
except ImportError:
    print("✗ pywt 不可用")