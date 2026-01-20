cmake_minimum_required(VERSION 3.18)
project(YoloQuantizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Dependency Paths ---
# If your libraries are in non-standard locations, you may need to adjust these paths.
# For example: set(TENSORRT_ROOT "/path/to/your/TensorRT-10.x.x.x")
set(TENSORRT_ROOT "/workspace/churui/TensorRT-10.10.0.31" CACHE PATH "Root directory of the TensorRT installation")
set(OpenCV_DIR "/usr/include/opencv4" CACHE PATH "Path to OpenCV cmake config")

# --- Find CUDA ---
# This requires CUDA to be in the system's PATH or CMAKE_PREFIX_PATH
find_package(CUDA 11.0 REQUIRED)
message(STATUS "Found CUDA: ${CUDA_VERSION_STRING}")
set(CMAKE_CUDA_STANDARD 17)

# --- Find OpenCV ---
# This is needed for reading and preprocessing calibration images.
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found. Please install it (e.g., sudo apt-get install libopencv-dev) or set OpenCV_DIR.")
endif()

# --- Find TensorRT ---
# Adjust paths if your installation differs from standard locations.
if (NOT EXISTS "${TENSORRT_ROOT}")
    message(FATAL_ERROR "TensorRT root directory not found at ${TENSORRT_ROOT}. Please set the TENSORRT_ROOT variable.")
endif()

# TensorRT include directories
include_directories(
    SYSTEM ${CUDA_INCLUDE_DIRS}
    SYSTEM ${TENSORRT_ROOT}/include
)

# On some systems, TRT headers are installed in a separate location.
# For Debian/Ubuntu, they are often in /usr/include/x86_64-linux-gnu/
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu/
    NO_DEFAULT_PATH
)
if(TRT_INCLUDE_DIR)
    include_directories(SYSTEM ${TRT_INCLUDE_DIR})
    message(STATUS "Found TensorRT headers at: ${TRT_INCLUDE_DIR}")
else()
    message(WARNING "Did not find TensorRT headers in /usr/include/x86_64-linux-gnu/. Assuming they are in ${TENSORRT_ROOT}/include")
endif()


# TensorRT library files
# For Debian/Ubuntu, they are often in /usr/lib/x86_64-linux-gnu/
find_library(NVINFER_LIB nvinfer NAMES libnvinfer.so PATHS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu/)
find_library(NVONNXPARSER_LIB nvonnxparser NAMES libnvonnxparser.so PATHS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu/)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin NAMES libnvinfer_plugin.so PATHS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu/)

if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "Could not find TensorRT libraries (nvinfer, nvonnxparser). Check TENSORRT_ROOT or system library paths.")
endif()

message(STATUS "Found TensorRT libs: ${NVINFER_LIB}; ${NVONNXPARSER_LIB}")

# --- Add Executable ---
add_executable(quantize_yolo src/main.cpp)

# --- Link Libraries ---
target_link_libraries(quantize_yolo
    PRIVATE
    # CUDA
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}

    # TensorRT
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${NVINFER_PLUGIN_LIB}

    # OpenCV
    ${OpenCV_LIBS}
)

# --- Build Instructions ---
message(STATUS "Project configured. To build, run:")
message(STATUS "  mkdir build")
message(STATUS "  cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "\nTo run the executable:")
message(STATUS "  ./quantize_yolo")
