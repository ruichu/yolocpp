# YOLO C++ 推理与量化

基于 TensorRT 的 YOLO 模型推理和 INT8 量化统一项目。支持多个 YOLO 版本（YOLO v8 和 YOLO v26），自动检测模型类型。

## 功能特性

- **YOLO 推理**: 在 TensorRT engine 文件上运行推理
- **INT8 量化**: 将 ONNX 模型转换为 INT8 TensorRT engine
- **多版本支持**: 自动检测并处理 YOLO v8 和 YOLO v26 格式
- **灵活的 TensorRT 检测**: 使用 `TRT_ROOT` 环境变量或系统安装的 TensorRT
- **可配置参数**: 通过命令行参数配置量化设置

## 项目结构

```
.
├── include/              # 公共头文件
│   ├── common.h         # 共享类型和常量
│   ├── trt_logger.h     # TensorRT 日志类
│   ├── int8_calibrator.h # INT8 校准器
│   ├── cudaHelper.hpp   # CUDA 工具类
│   └── timeTest.hpp     # 计时工具
├── src/                  # 源文件
│   ├── yolo_infer.cpp   # 推理程序
│   └── yolo_quantize.cpp # 量化程序
├── CMakeLists.txt       # 统一构建配置
└── README.md
```

## 依赖项

- **CUDA** 11.0+
- **TensorRT** 8.x 或 10.x
- **OpenCV** 4.x
- **C++17** 兼容编译器

## 安装

### 1. 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit libopencv-dev

# 安装 TensorRT（参考 NVIDIA 官方指南）
# https://developer.nvidia.com/tensorrt
```

### 2. 设置 TensorRT 路径（可选）

如果 TensorRT 没有安装在系统路径下，设置 `TRT_ROOT` 环境变量：

```bash
export TRT_ROOT=/path/to/your/TensorRT
```

注意：同时也支持 `TENSORRT_ROOT` 以保持向后兼容。

### 3. 构建项目

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

## 使用方法

### YOLO 推理

在 TensorRT engine 文件上运行推理：

```bash
./yolo_infer <engine_file> <input_image> [conf_thresh] [nms_thresh]
```

**示例：**
```bash
./yolo_infer model.engine image.jpg 0.5 0.5
```

**参数说明：**
- `engine_file`: TensorRT engine 文件路径
- `input_image`: 输入图像路径
- `conf_thresh`: 置信度阈值（默认：0.5）
- `nms_thresh`: NMS 阈值（默认：0.5）

### YOLO 量化

将 ONNX 模型转换为 INT8 TensorRT engine：

```bash
./yolo_quantize <onnx_file> <output_engine> [options]
```

**示例：**
```bash
./yolo_quantize yolo.onnx yolo_int8.engine --calib-path ./calib_images --batch-size 8
```

**参数说明：**
- `onnx_file`: 输入 ONNX 模型路径
- `output_engine`: 输出 TensorRT engine 路径

**可选参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--calib-path PATH` | 校准图像目录路径 | `./calibration_data` |
| `--cache-file FILE` | 校准缓存文件 | `yolo.cache` |
| `--batch-size N` | 校准批次大小 | `8` |
| `--input-width W` | 输入图像宽度 | `640` |
| `--input-height H` | 输入图像高度 | `640` |
| `--input-name NAME` | 输入张量名称 | `images` |
| `--workspace SIZE` | 最大工作空间大小 (MB) | `1024` |
| `--help` | 显示帮助信息 | - |

## 模型类型检测

项目根据输出张量形状自动检测 YOLO 版本：

- **YOLO v8**: 输出形状 `(1, 84, 8400)` 或类似 - 需要 NMS 后处理
- **YOLO v26**: 输出形状 `(1, 300, 6)` - NMS-free 格式

## 重构说明

这是原项目重构后的版本，主要变更：

1. **统一构建**: 单一 CMake 项目，生成两个可执行文件
2. **文件重命名**: `yolo11.cpp` → `src/yolo_infer.cpp`
3. **Bug 修复**: YOLO v26 检测现在正确检查 `(1, 300, 6)` 而不是 `(300, 6)`
4. **灵活的 TensorRT**: `TRT_ROOT` 现在是可选的；未设置时使用系统默认路径
5. **通用命名**: 移除了 "yolo11" 相关命名；现在支持多个 YOLO 版本
6. **可配置量化**: 参数通过命令行传递，不再硬编码

## 已废弃的文件

以下文件在迁移后已被移除：
- `yolo11.cpp`（已替换为 `src/yolo_infer.cpp`）
- `Makefile`（已替换为 `CMakeLists.txt`）
- `quantize/` 目录（代码已整合到主项目）

## 许可证

本项目使用 TensorRT，遵循 NVIDIA 的许可条款。
