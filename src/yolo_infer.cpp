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
#include <algorithm>

#include "cudaHelper.hpp"
#include "timeTest.hpp"
#include "trt_logger.h"
#include "common.h"

// Default parameters
const YoloParams DEFAULT_PARAMS = {
    .inputHeight = 640,
    .inputWidth = 640,
    .confThresh = 0.5f,
    .nmsThresh = 0.5f
};

TEST_TIME_BEGIN(postProcess)

// ===================== Preprocessing =====================
// Preprocess image for YOLO inference
// Returns CHW format normalized float vector
static std::vector<float> preprocessImage(const cv::Mat& img, const YoloParams& params)
{
    cv::Mat resizedImg, rgbImg;
    // Resize to input dimensions
    cv::resize(img, resizedImg, cv::Size(params.inputWidth, params.inputHeight));
    // BGR to RGB
    cv::cvtColor(resizedImg, rgbImg, cv::COLOR_BGR2RGB);
    // Normalize to 0-1
    rgbImg.convertTo(rgbImg, CV_32FC3, 1.0 / 255.0);

    // Prepare host input buffer (CHW format)
    std::vector<float> hostInput(3 * params.inputWidth * params.inputHeight);

    // HWC -> CHW
    int idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < params.inputHeight; ++h)
            for (int w = 0; w < params.inputWidth; ++w)
                hostInput[idx++] = rgbImg.at<cv::Vec3f>(h, w)[c];

    return hostInput;
}

// ===================== Postprocessing =====================
// Detect model type from output tensor dimensions
static ModelType detectModelType(const nvinfer1::Dims& outputDims)
{
    // YOLO v26 output shape: (1, 300, 6) - batch_size=1, num_dets=300, 6=[x1,y1,x2,y2,score,class_id]
    // YOLO v8 output shape: (1, 84, 8400) or similar - requires NMS

    // Check for YOLO v26 (NMS-free) format: (1, 300, 6)
    if (outputDims.nbDims == 3 &&
        outputDims.d[0] == 1 &&
        outputDims.d[1] == 300 &&
        outputDims.d[2] == 6)
    {
        return ModelType::YOLO_V26;
    }

    // Check for YOLO v8 format - look for 8400 anchors (last dimension or second dimension)
    if (outputDims.nbDims >= 2)
    {
        // Format like (1, 84, 8400) or (84, 8400)
        if (outputDims.d[outputDims.nbDims - 1] == 8400 ||
            (outputDims.nbDims >= 2 && outputDims.d[1] == 8400))
        {
            return ModelType::YOLO_V8;
        }
    }

    return ModelType::UNKNOWN;
}

// Get model type name as string
static std::string getModelTypeName(ModelType type)
{
    switch (type)
    {
        case ModelType::YOLO_V8: return "YOLO v8 (legacy, requires NMS)";
        case ModelType::YOLO_V26: return "YOLO v26 (NMS-free)";
        default: return "Unknown (defaulting to YOLO v8)";
    }
}

// NMS (Non-Maximum Suppression)
static std::vector<DetectBox> nms(std::vector<DetectBox>& boxes, float nms_thresh)
{
    std::vector<DetectBox> result;
    std::sort(boxes.begin(), boxes.end(), [](const DetectBox& a, const DetectBox& b)
              { return a.conf > b.conf; });

    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); i++)
    {
        if (suppressed[i])
            continue;
        result.push_back(boxes[i]);
        float area_i = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
        for (size_t j = i + 1; j < boxes.size(); j++)
        {
            if (suppressed[j])
                continue;
            float x1 = std::max(boxes[i].x1, boxes[j].x1);
            float y1 = std::max(boxes[i].y1, boxes[j].y1);
            float x2 = std::min(boxes[i].x2, boxes[j].x2);
            float y2 = std::min(boxes[i].y2, boxes[j].y2);
            float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float iou = inter_area / (area_i + (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1) - inter_area);
            if (iou > nms_thresh)
                suppressed[j] = true;
        }
    }
    return result;
}

// YOLO v8 post-processing (with NMS)
static std::vector<DetectBox> postprocessYOLOv8(const cv::Mat& img,
                                                  const std::vector<float>& output_vector,
                                                  const YoloParams& params)
{
    float scale = std::min(static_cast<float>(params.inputWidth) / img.cols,
                           static_cast<float>(params.inputHeight) / img.rows);
    float pad_w = (params.inputWidth - img.cols * scale) / 2.0f;
    float pad_h = (params.inputHeight - img.rows * scale) / 2.0f;

    std::vector<DetectBox> boxes;
    size_t num_classes = COCO_CLASS_NAMES.size();
    int elements_per_box = 4 + num_classes;

    // Reshape 1D array to (84, 8400) then transpose to (8400, 84)
    cv::Mat output_mat_1d(1, output_vector.size(), CV_32F, const_cast<float*>(output_vector.data()));
    cv::Mat output_mat_2d = output_mat_1d.reshape(1, elements_per_box);
    cv::Mat output;
    cv::transpose(output_mat_2d, output);

    // Split bbox coordinates and class probabilities
    cv::Mat bbox_coords = output.colRange(0, 4).clone();                // (8400, 4) [cx, cy, w, h]
    cv::Mat class_probs = output.colRange(4, elements_per_box).clone(); // (8400, 80) class probs

    // Find max confidence and class ID for each anchor
    for (int i = 0; i < output.rows; ++i)
    {
        cv::Mat row_probs = class_probs.row(i);
        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(row_probs, nullptr, &max_val, nullptr, &max_loc);

        if (max_val < params.confThresh)
            continue;

        DetectBox box;
        float cx = bbox_coords.at<float>(i, 0);
        float cy = bbox_coords.at<float>(i, 1);
        float w = bbox_coords.at<float>(i, 2);
        float h = bbox_coords.at<float>(i, 3);
        box.x1 = (cx - w / 2.0f - pad_w) / scale;
        box.y1 = (cy - h / 2.0f - pad_h) / scale;
        box.x2 = (cx + w / 2.0f - pad_w) / scale;
        box.y2 = (cy + h / 2.0f - pad_h) / scale;
        box.conf = static_cast<float>(max_val);
        box.class_id = max_loc.x;
        box.class_name = getClassName(max_loc.x);
        boxes.push_back(box);
    }

    return nms(boxes, params.nmsThresh);
}

// YOLO v26 post-processing (NMS-free)
static std::vector<DetectBox> postprocessYOLOv26(const cv::Mat& img,
                                                  const std::vector<float>& output_vector,
                                                  const YoloParams& params)
{
    std::vector<DetectBox> boxes;

    // YOLO v26 output format: (1, 300, 6) = [x1, y1, x2, y2, score, class_id]
    // Output coordinates are already on input scale (640x640)

    float scale = std::min(static_cast<float>(params.inputWidth) / img.cols,
                           static_cast<float>(params.inputHeight) / img.rows);
    float pad_w = (params.inputWidth - img.cols * scale) / 2.0f;
    float pad_h = (params.inputHeight - img.rows * scale) / 2.0f;

    int num_detections = 300;
    int elements_per_detection = 6;  // x1, y1, x2, y2, score, class_id

    for (int i = 0; i < num_detections; ++i)
    {
        int base_idx = i * elements_per_detection;
        if (base_idx + elements_per_detection > static_cast<int>(output_vector.size()))
            break;

        float x1 = output_vector[base_idx + 0];
        float y1 = output_vector[base_idx + 1];
        float x2 = output_vector[base_idx + 2];
        float y2 = output_vector[base_idx + 3];
        float conf = output_vector[base_idx + 4];
        int class_id = static_cast<int>(output_vector[base_idx + 5]);

        // Skip low confidence detections
        if (conf < params.confThresh)
            continue;

        DetectBox box;
        box.x1 = (x1 - pad_w) / scale;
        box.y1 = (y1 - pad_h) / scale;
        box.x2 = (x2 - pad_w) / scale;
        box.y2 = (y2 - pad_h) / scale;
        box.conf = conf;
        box.class_id = class_id;
        box.class_name = getClassName(class_id);
        boxes.push_back(box);
    }

    return boxes;
}

// Unified post-processing entry point (auto-detects model type)
static std::vector<DetectBox> postprocess(const cv::Mat& img,
                                           const std::vector<float>& output_vector,
                                           ModelType modelType,
                                           const YoloParams& params)
{
    switch (modelType)
    {
    case ModelType::YOLO_V8:
        std::cout << "Using YOLO v8 post-processing (with NMS)" << std::endl;
        return postprocessYOLOv8(img, output_vector, params);
    case ModelType::YOLO_V26:
        std::cout << "Using YOLO v26 post-processing (NMS-free)" << std::endl;
        return postprocessYOLOv26(img, output_vector, params);
    default:
        std::cerr << "Warning: Unknown model type, defaulting to YOLO v8" << std::endl;
        return postprocessYOLOv8(img, output_vector, params);
    }
}

// ===================== Inference =====================
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
