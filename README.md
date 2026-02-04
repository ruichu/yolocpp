# YOLO C++ Inference and Quantization

A unified C++ project for YOLO model inference and INT8 quantization using TensorRT. Supports multiple YOLO versions (YOLO v8 and YOLO v26) with automatic model type detection.

## Features

- **YOLO Inference**: Run inference on TensorRT engine files
- **INT8 Quantization**: Convert ONNX models to INT8 TensorRT engines
- **Multi-version Support**: Automatically detects and handles YOLO v8 and YOLO v26 formats
- **Flexible TensorRT Detection**: Uses `TRT_ROOT` environment variable or system-installed TensorRT
- **Configurable Parameters**: Command-line arguments for quantization settings

## Project Structure

```
.
├── include/              # Common headers
│   ├── common.h         # Shared types and constants
│   ├── postprocess.h    # Post-processing logic
│   ├── preprocess.h     # Image preprocessing
│   ├── trt_logger.h     # TensorRT logger
│   ├── int8_calibrator.h # INT8 calibrator
│   ├── cudaHelper.hpp   # CUDA utilities
│   └── timeTest.hpp     # Timing utilities
├── src/                  # Source files
│   ├── yolo_infer.cpp   # Inference executable
│   └── yolo_quantize.cpp # Quantization executable
├── CMakeLists.txt       # Unified build configuration
└── README.md
```

## Dependencies

- **CUDA** 11.0+
- **TensorRT** 8.x or 10.x
- **OpenCV** 4.x
- **C++17** compatible compiler

## Installation

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit libopencv-dev

# Install TensorRT (follow NVIDIA's official guide)
# https://developer.nvidia.com/tensorrt
```

### 2. Set TensorRT Path (Optional)

If TensorRT is not installed in a system location, set the `TRT_ROOT` environment variable:

```bash
export TRT_ROOT=/path/to/your/TensorRT
```

Note: `TENSORRT_ROOT` is also supported for legacy compatibility.

### 3. Build the Project

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

## Usage

### YOLO Inference

Run inference on a TensorRT engine file:

```bash
./yolo_infer <engine_file> <input_image> [conf_thresh] [nms_thresh]
```

**Example:**
```bash
./yolo_infer model.engine image.jpg 0.5 0.5
```

**Arguments:**
- `engine_file`: Path to TensorRT engine file
- `input_image`: Path to input image
- `conf_thresh`: Confidence threshold (default: 0.5)
- `nms_thresh`: NMS threshold (default: 0.5)

### YOLO Quantization

Convert ONNX model to INT8 TensorRT engine:

```bash
./yolo_quantize <onnx_file> <output_engine> [options]
```

**Example:**
```bash
./yolo_quantize yolo.onnx yolo_int8.engine --calib-path ./calib_images --batch-size 8
```

**Arguments:**
- `onnx_file`: Path to input ONNX model
- `output_engine`: Path for output TensorRT engine

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--calib-path PATH` | Path to calibration images | `./calibration_data` |
| `--cache-file FILE` | Calibration cache file | `yolo.cache` |
| `--batch-size N` | Calibration batch size | `8` |
| `--input-width W` | Input image width | `640` |
| `--input-height H` | Input image height | `640` |
| `--input-name NAME` | Input tensor name | `images` |
| `--workspace SIZE` | Max workspace size (MB) | `1024` |
| `--help` | Show help message | - |

## Model Type Detection

The project automatically detects the YOLO version based on the output tensor shape:

- **YOLO v8**: Output shape `(1, 84, 8400)` or similar - requires NMS post-processing
- **YOLO v26**: Output shape `(1, 300, 6)` - NMS-free format

## Migration Notes

This is a refactored version of the original project. Key changes:

1. **Unified Build**: Single CMake project with two executables
2. **Renamed Files**: `yolo11.cpp` → `src/yolo_infer.cpp`
3. **Bug Fix**: YOLO v26 detection now correctly checks `(1, 300, 6)` instead of `(300, 6)`
4. **Flexible TensorRT**: `TRT_ROOT` is now optional; system defaults are used if not set
5. **Generic Naming**: Removed "yolo11" references; now supports multiple YOLO versions
6. **Configurable Quantization**: Parameters passed via command-line instead of hardcoded

## Old Files (Deprecated)

The following files can be removed after migration:
- `yolo11.cpp` (replaced by `src/yolo_infer.cpp`)
- `Makefile` (replaced by `CMakeLists.txt`)
- `quantize/` directory (code integrated into main project)

## License

This project uses TensorRT and follows NVIDIA's licensing terms.
