cmake_minimum_required(VERSION 3.18)
project(YoloInfer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==================== Options ====================
option(USE_SYSTEM_TENSORRT "Use system-installed TensorRT instead of TRT_ROOT" OFF)

# ==================== TensorRT Configuration ====================
# Check if TRT_ROOT environment variable is set
if(DEFINED ENV{TRT_ROOT})
    set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "Root directory of the TensorRT installation")
    message(STATUS "Using TRT_ROOT from environment: ${TRT_ROOT}")
elseif(DEFINED ENV{TENSORRT_ROOT})
    # Legacy support: TENSORRT_ROOT -> TRT_ROOT
    set(TRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "Root directory of the TensorRT installation")
    message(STATUS "Using TENSORRT_ROOT from environment (mapped to TRT_ROOT): ${TRT_ROOT}")
elseif(NOT USE_SYSTEM_TENSORRT)
    # Try to find TensorRT in common locations
    set(TRT_ROOT_SEARCH_PATHS
        "/usr/local/TensorRT"
        "/opt/TensorRT"
        "C:/TensorRT"
        "C:/Program Files/TensorRT"
    )

    foreach(SEARCH_PATH ${TRT_ROOT_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}")
            set(TRT_ROOT "${SEARCH_PATH}" CACHE PATH "Root directory of the TensorRT installation")
            message(STATUS "Found TensorRT at: ${TRT_ROOT}")
            break()
        endif()
    endforeach()
endif()

# If TRT_ROOT is still not set, use system TensorRT
if(NOT DEFINED TRT_ROOT OR NOT EXISTS "${TRT_ROOT}")
    message(STATUS "TRT_ROOT not set or path does not exist, using system TensorRT")
    set(USE_SYSTEM_TENSORRT ON)
endif()

# ==================== CUDA ====================
find_package(CUDA 11.0 QUIET)
if(CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION_STRING}")
    set(CMAKE_CUDA_STANDARD 17)
else()
    message(WARNING "CUDA not found via find_package. Trying pkg-config...")
    # Try to find CUDA via pkg-config
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(CUDA cuda)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA via pkg-config: ${CUDA_VERSION}")
        endif()
    endif()
endif()

# ==================== OpenCV ====================
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION} (${OpenCV_DIR})")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV development libraries.")
endif()

# ==================== TensorRT ====================
if(USE_SYSTEM_TENSORRT)
    # Use system-installed TensorRT
    message(STATUS "Searching for system-installed TensorRT...")

    # Check common system paths
    set(TRT_SYSTEM_INCLUDE_PATHS
        "/usr/include"
        "/usr/local/include"
        "/usr/include/x86_64-linux-gnu"
    )

    foreach(INC_PATH ${TRT_SYSTEM_INCLUDE_PATHS})
        if(EXISTS "${INC_PATH}/NvInfer.h")
            set(TRT_INCLUDE_DIR "${INC_PATH}")
            message(STATUS "Found TensorRT headers at: ${TRT_INCLUDE_DIR}")
            break()
        endif()
    endforeach()

    if(NOT DEFINED TRT_INCLUDE_DIR)
        message(FATAL_ERROR "TensorRT headers not found in system paths. "
                            "Please install TensorRT or set TRT_ROOT environment variable.")
    endif()

    # Find libraries in system paths
    find_library(NVINFER_LIB nvinfer)
    find_library(NVONNXPARSER_LIB nvonnxparser)
    find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)

else()
    # Use TRT_ROOT
    if(NOT EXISTS "${TRT_ROOT}")
        message(FATAL_ERROR "TRT_ROOT is set but path does not exist: ${TRT_ROOT}")
    endif()

    set(TRT_INCLUDE_DIR "${TRT_ROOT}/include")
    set(TRT_LIB_DIR "${TRT_ROOT}/lib")

    message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
    message(STATUS "TensorRT lib: ${TRT_LIB_DIR}")

    # Find libraries
    find_library(NVINFER_LIB nvinfer PATHS ${TRT_LIB_DIR} NO_DEFAULT_PATH)
    find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TRT_LIB_DIR} NO_DEFAULT_PATH)
    find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS ${TRT_LIB_DIR} NO_DEFAULT_PATH)
endif()

# Validate TensorRT libraries were found
if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "Could not find required TensorRT libraries (nvinfer, nvonnxparser). "
                        "Please install TensorRT or set TRT_ROOT correctly.")
endif()

message(STATUS "TensorRT libraries: ${NVINFER_LIB}, ${NVONNXPARSER_LIB}")

# ==================== Include Directories ====================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)

# ==================== Common Libraries ====================
set(COMMON_LIBS
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${OpenCV_LIBS}
)

if(NVINFER_PLUGIN_LIB)
    list(APPEND COMMON_LIBS ${NVINFER_PLUGIN_LIB})
endif()

# ==================== Build Targets ====================

# --- yolo_infer executable ---
add_executable(yolo_infer src/yolo_infer.cpp)
target_link_libraries(yolo_infer PRIVATE ${COMMON_LIBS})

# --- yolo_quantize executable ---
add_executable(yolo_quantize src/yolo_quantize.cpp)
target_link_libraries(yolo_quantize PRIVATE ${COMMON_LIBS})

# ==================== Installation ====================
install(TARGETS yolo_infer yolo_quantize
    RUNTIME DESTINATION bin
)

# ==================== Summary ====================
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "TensorRT mode:        ${USE_SYSTEM_TENSORRT}")
message(STATUS "OpenCV version:       ${OpenCV_VERSION}")
message(STATUS "========================")
message(STATUS "")
message(STATUS "To build, run:")
message(STATUS "  mkdir -p build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  cmake --build .")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  yolo_infer    - Run YOLO inference on an engine file")
message(STATUS "  yolo_quantize - Quantize ONNX model to INT8 engine")
message(STATUS "")
