# 定义编译器
CXX = g++

# -------------------------- 路径配置 --------------------------
# 从环境变量 TRT_ROOT 读取 TensorRT 根目录
# 如果未设置环境变量，会给出明确提示
ifndef TRT_ROOT
$(error 环境变量 TRT_ROOT 未设置，请先执行 export TRT_ROOT=/path/to/your/TensorRT-8.x.x.x)
endif

# CUDA 的根目录 (通常是 /usr/local/cuda，若不同可通过环境变量 CUDA_ROOT 覆盖)
CUDA_DIR ?= /usr/local/cuda
# ---------------------------------------------------------------------------------------

# 可执行文件名称
TARGET = yolo11_infer

# 源文件
SRC = yolo11.cpp cudaHelper.hpp

# 头文件包含路径 (直接使用环境变量 TRT_ROOT)
INCLUDES = -I$(TRT_ROOT)/include \
           -I$(CUDA_DIR)/include \
		   -I/usr/include/opencv4

# 库文件链接路径
LDFLAGS = -L$(TRT_ROOT)/lib \
          -L$(CUDA_DIR)/lib64

# 需要链接的库
LIBS = -lnvinfer \
       -lnvonnxparser \
       -lcudart \
       `pkg-config --libs opencv4`

# 编译选项 (TensorRT 需 C++11 及以上)
CXXFLAGS = -std=c++11 -Wall

# 默认目标：编译生成可执行文件
all: $(TARGET)

# 编译规则
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(LDFLAGS) $(LIBS)

# 清理编译产物
clean:
	rm -rf $(TARGET)