// ===================== Disable TensorRT deprecation warnings locally =====================
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <NvInferRuntime.h>
#pragma GCC diagnostic pop

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "cudaHelper.hpp"
#include "timeTest.hpp"
#include "trt_logger.h"
#include "common.h"
#include "preprocess.h"
#include "postprocess.h"

// Default parameters
const YoloParams DEFAULT_PARAMS = {
    .inputHeight = 640,
    .inputWidth = 640,
    .confThresh = 0.5f,
    .nmsThresh = 0.5f
};

TEST_TIME_BEGIN(postProcess)

// Load .engine file to memory
std::vector<char> loadEngineFile(const std::string& enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open engine file: " + enginePath);
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    return engineData;
}

// Inference result structure
struct InferenceResult
{
    std::vector<float> outputData;
    ModelType modelType;
    std::string modelTypeName;
};

// Core inference function
InferenceResult runInference(const std::string& enginePath, const std::vector<float>& inputData)
{
    InferenceResult result;

    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime(
        nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        throw std::runtime_error("Failed to create TensorRT Runtime");
    }

    // Load engine file and deserialize
    std::vector<char> engineData = loadEngineFile(enginePath);
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine)
    {
        throw std::runtime_error("Failed to deserialize CUDA Engine");
    }

    TEST_TIME_BEGIN(preInfer)

    // Create execution context
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context(
        engine->createExecutionContext());
    if (!context)
    {
        throw std::runtime_error("Failed to create Execution Context");
    }

    // Get input/output tensor info
    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);

    // Get tensor shapes
    nvinfer1::Dims inputDims = engine->getTensorShape(inputName);
    nvinfer1::Dims outputDims = engine->getTensorShape(outputName);

    // Print output tensor shape for debugging
    std::cout << "Output tensor shape: [";
    for (int i = 0; i < outputDims.nbDims; ++i)
    {
        std::cout << outputDims.d[i];
        if (i < outputDims.nbDims - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Detect model type
    result.modelType = detectModelType(outputDims);
    result.modelTypeName = getModelTypeName(result.modelType);
    std::cout << "Detected model: " << result.modelTypeName << std::endl;

    auto calculateElementCount = [](const nvinfer1::Dims& dims)
    {
        size_t count = 1;
        for (auto i = 0; i < dims.nbDims; i++)
        {
            count *= dims.d[i];
        }
        return count;
    };
    size_t inputElementCount = calculateElementCount(inputDims);
    size_t outputElementCount = calculateElementCount(outputDims);

    // Validate input data length
    if (inputData.size() != inputElementCount)
    {
        throw std::runtime_error(
            "Input data size mismatch: expected " + std::to_string(inputElementCount) +
            ", got " + std::to_string(inputData.size()));
    }

    // Allocate GPU memory
    CudaBuffer<float> dInput(inputElementCount), dOutput(outputElementCount);
    if (dInput.empty() || dOutput.empty())
    {
        throw std::runtime_error("Failed to allocate device memory");
    }

    // Set tensor addresses (TensorRT 10.x recommends setTensorAddress)
    context->setTensorAddress(inputName, dInput.get());
    context->setTensorAddress(outputName, dOutput.get());

    // Copy input data to GPU
    dInput.copyFromHost(inputData.data(), inputElementCount);

    TEST_TIME_END(preInfer)

    TEST_TIME_BEGIN(infer)

    // Execute inference
    CudaStream stream;
    bool success = context->enqueueV3(stream.get());
    if (!success)
    {
        throw std::runtime_error("Inference enqueue failed");
    }

    // Wait for stream completion
    stream.synchronize();

    TEST_TIME_END(infer)

    UPDATE_TEST_TIME(postProcess)

    result.outputData.resize(outputElementCount);
    dOutput.copyToHost(result.outputData.data(), outputElementCount);

    return result;
}

// Main function
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <path-to-engine-file> <path-to-input-image>" << std::endl;
        std::cerr << "Supports both YOLO v8 and YOLO v26 models (auto-detected)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Optional arguments:" << std::endl;
        std::cerr << "  [conf-thresh]  Confidence threshold (default: 0.5)" << std::endl;
        std::cerr << "  [nms-thresh]   NMS threshold (default: 0.5)" << std::endl;
        return 1;
    }

    try
    {
        // Parse optional arguments
        YoloParams params = DEFAULT_PARAMS;
        if (argc >= 4)
        {
            params.confThresh = std::stof(argv[3]);
        }
        if (argc >= 5)
        {
            params.nmsThresh = std::stof(argv[4]);
        }

        TEST_TIME_BEGIN(loadImage)

        // Load and preprocess input image
        cv::Mat img = cv::imread(argv[2]);
        if (img.empty())
        {
            throw std::runtime_error("Failed to read input image: " + std::string(argv[2]));
        }

        TEST_TIME_END(loadImage)

        TEST_TIME_BEGIN(preprocessImage)
        std::vector<float> inputData = preprocessImage(img, params);
        TEST_TIME_END(preprocessImage)

        InferenceResult inferResult = runInference(argv[1], inputData);

        auto detectResults = postprocess(img, inferResult.outputData, inferResult.modelType, params);

        TEST_TIME_END(postProcess)

        std::cout << "\n=== Detection Results (" << detectResults.size() << " objects) ===" << std::endl;
        for (size_t i = 0; i < detectResults.size(); i++)
        {
            const auto& box = detectResults[i];
            std::cout << "[" << i + 1 << "] " << box.class_name << " (ID: " << box.class_id << ") "
                      << "Conf: " << box.conf << " "
                      << "Box: [" << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << "]"
                      << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
