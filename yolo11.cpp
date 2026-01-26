// ===================== 局部禁用TensorRT的废弃警告 =====================
// 1. 保存当前警告配置
#pragma GCC diagnostic push
// 2. 临时忽略废弃声明警告（仅对后续TensorRT头文件生效）
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <NvInferRuntime.h>
// 3. 恢复警告配置（你的业务代码仍会检测警告）
#pragma GCC diagnostic pop

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "cudaHelper.hpp"
#include "timeTest.hpp"

const int INPUT_HEIGHT = 640;
const int INPUT_WIDTH = 640;
const float CONF_THRESH = 0.5;
const float NMS_THRESH = 0.5;

TEST_TIME_BEGIN(postProcess)

const std::vector<std::string> CLASS_NAMES =
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// 模型类型枚举
enum class ModelType
{
    UNKNOWN,
    YOLO11,  // 旧格式：输出 (8400, 84)，需要 NMS
    YOLO26   // 新格式：输出 (300, 6)，NMS-free
};

// 自定义日志类，用于输出 TensorRT 运行时日志
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        // 只输出 WARNING 及以上级别的日志
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "TensorRT Log (" << static_cast<int>(severity) << "): " << msg << std::endl;
        }
    }
};

// 智能指针释放器（TensorRT 10.x 兼容）
template <typename T>
struct Deleter
{
    void operator()(T *ptr) const
    {
        if (ptr)
        {
            delete ptr; // ptr->destroy();
        }
    }
};

// 检测框结构体
struct DetectBox
{
    float x1, y1, x2, y2;
    float conf;
    int class_id;
    std::string class_name;
};

// 预处理图片（BGR→RGB、归一化、转CHW、复制到Device）
std::vector<float> preprocessImage(const cv::Mat &img)
{
    cv::Mat resizedImg, rgbImg;
    // 缩放至输入尺寸
    cv::resize(img, resizedImg, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    // BGR→RGB
    cv::cvtColor(resizedImg, rgbImg, cv::COLOR_BGR2RGB);
    // 归一化到0~1
    rgbImg.convertTo(rgbImg, CV_32FC3, 1.0 / 255.0);

    // 准备Host输入缓冲区
    std::vector<float> hostInput(3 * INPUT_WIDTH * INPUT_HEIGHT);

    // HWC→CHW（YOLOv8要求CHW格式）
    int idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < INPUT_HEIGHT; ++h)
            for (int w = 0; w < INPUT_WIDTH; ++w)
                hostInput[idx++] = rgbImg.at<cv::Vec3f>(h, w)[c];

    return hostInput;
}

// NMS非极大值抑制
std::vector<DetectBox> nms(std::vector<DetectBox> &boxes)
{
    std::vector<DetectBox> result;
    std::sort(boxes.begin(), boxes.end(), [](const DetectBox &a, const DetectBox &b)
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
            if (iou > NMS_THRESH)
                suppressed[j] = true;
        }
    }
    return result;
}

// 根据输出张量形状自动检测模型类型
ModelType detectModelType(const nvinfer1::Dims &outputDims)
{
    // YOLO26 输出形状: (300, 6) - batch_size=1, num_dets=300, 6=[x1,y1,x2,y2,score,class_id]
    // YOLO11 输出形状: (1, 84, 8400) 或 (84, 8400)

    if (outputDims.nbDims == 2 && outputDims.d[0] == 300 && outputDims.d[1] == 6)
    {
        return ModelType::YOLO26;
    }
    else if (outputDims.nbDims >= 2 && outputDims.d[1] == 8400)
    {
        return ModelType::YOLO11;
    }
    else if (outputDims.nbDims >= 2 && (outputDims.d[outputDims.nbDims - 1] == 8400 ||
                                      outputDims.d[0] == 8400))
    {
        return ModelType::YOLO11;
    }

    return ModelType::UNKNOWN;
}

// YOLO11 后处理：解析输出+还原坐标
std::vector<DetectBox> postprocessYOLO11(const cv::Mat &img, const std::vector<float> &output_vector)
{
    float scale = std::min(static_cast<float>(INPUT_WIDTH) / img.cols, static_cast<float>(INPUT_HEIGHT) / img.rows);
    float pad_w = (INPUT_WIDTH - img.cols * scale) / 2.0f;
    float pad_h = (INPUT_HEIGHT - img.rows * scale) / 2.0f;

    std::vector<DetectBox> boxes;
    size_t num_classes = CLASS_NAMES.size();
    int elements_per_box = 4 + num_classes;

    // 第一步：将一维数组转为cv::Mat，初始维度为 (1, 84*8400)
    cv::Mat output_mat_1d(1, output_vector.size(), CV_32F, const_cast<float *>(output_vector.data()));

    // 第二步：重塑为 (84, 8400)
    cv::Mat output_mat_2d = output_mat_1d.reshape(1, elements_per_box);

    // 转置为 (8400, 84)
    cv::Mat output;
    cv::transpose(output_mat_2d, output);

    // 分离边界框坐标和类别概率
    cv::Mat bbox_coords = output.colRange(0, 4).clone();                // (8400, 4) [cx, cy, w, h]
    cv::Mat class_probs = output.colRange(4, elements_per_box).clone(); // (8400, 80) 类别概率

    // 计算每个锚点的最大置信度和类别ID
    for (int i = 0; i < output.rows; ++i)
    {
        // 获取第i行的类别概率（80个值）
        cv::Mat row_probs = class_probs.row(i);
        double max_val;
        cv::Point max_loc;
        // 找该行的最大值和对应列索引（即类别ID）
        cv::minMaxLoc(row_probs, nullptr, &max_val, nullptr, &max_loc);

        if (max_val < CONF_THRESH)
            continue;

        DetectBox box;
        // 还原边界框坐标到原图尺寸
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
        if (static_cast<size_t>(max_loc.x) >= CLASS_NAMES.size())
            box.class_name = "unknown";
        else
            box.class_name = CLASS_NAMES[max_loc.x];
        boxes.push_back(box);
    }

    return nms(boxes);
}

// YOLO26 后处理：NMS-free 端到端输出
std::vector<DetectBox> postprocessYOLO26(const cv::Mat &img, const std::vector<float> &output_vector)
{
    std::vector<DetectBox> boxes;

    // YOLO26 输出格式: (300, 6) = [x1, y1, x2, y2, score, class_id]
    // 输出坐标已经在输入尺寸 (640x640) 上

    float scale = std::min(static_cast<float>(INPUT_WIDTH) / img.cols, static_cast<float>(INPUT_HEIGHT) / img.rows);
    float pad_w = (INPUT_WIDTH - img.cols * scale) / 2.0f;
    float pad_h = (INPUT_HEIGHT - img.rows * scale) / 2.0f;

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

        // 跳过低置信度检测（YOLO26 输出中无效检测的 score 可能是负数或0）
        if (conf < CONF_THRESH)
            continue;

        DetectBox box;
        box.x1 = (x1 - pad_w) / scale;
        box.y1 = (y1 - pad_h) / scale;
        box.x2 = (x2 - pad_w) / scale;
        box.y2 = (y2 - pad_h) / scale;
        box.conf = conf;
        box.class_id = class_id;
        if (static_cast<size_t>(class_id) >= CLASS_NAMES.size())
            box.class_name = "unknown";
        else
            box.class_name = CLASS_NAMES[class_id];
        boxes.push_back(box);
    }

    return boxes;
}

// 统一后处理入口函数（自动检测模型类型）
std::vector<DetectBox> postprocess(const cv::Mat &img, const std::vector<float> &output_vector, ModelType modelType)
{
    switch (modelType)
    {
    case ModelType::YOLO11:
        std::cout << "Using YOLO11 post-processing (with NMS)" << std::endl;
        return postprocessYOLO11(img, output_vector);
    case ModelType::YOLO26:
        std::cout << "Using YOLO26 post-processing (NMS-free)" << std::endl;
        return postprocessYOLO26(img, output_vector);
    default:
        std::cerr << "Warning: Unknown model type, defaulting to YOLO11" << std::endl;
        return postprocessYOLO11(img, output_vector);
    }
}

// 加载 .engine 文件到内存
std::vector<char> loadEngineFile(const std::string &enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open engine file: " + enginePath);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 读取文件内容
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    return engineData;
}

// 推理核心函数，返回输出数据和模型类型
struct InferenceResult
{
    std::vector<float> outputData;
    ModelType modelType;
    std::string modelTypeName;
};

InferenceResult runInference(const std::string &enginePath, const std::vector<float> &inputData)
{
    InferenceResult result;

    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime, Deleter<nvinfer1::IRuntime>> runtime(
        nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        throw std::runtime_error("Failed to create TensorRT Runtime");
    }

    // 加载 Engine 文件并反序列化
    std::vector<char> engineData = loadEngineFile(enginePath);
    std::unique_ptr<nvinfer1::ICudaEngine, Deleter<nvinfer1::ICudaEngine>> engine(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine)
    {
        throw std::runtime_error("Failed to deserialize CUDA Engine");
    }

    TEST_TIME_BEGIN(preInfer)

    // 创建执行上下文
    std::unique_ptr<nvinfer1::IExecutionContext, Deleter<nvinfer1::IExecutionContext>> context(
        engine->createExecutionContext());
    if (!context)
    {
        throw std::runtime_error("Failed to create Execution Context");
    }

    // 获取输入/输出张量信息（假设只有一个输入、一个输出）
    // 实际使用时可根据 Engine 配置调整
    const char *inputName = engine->getIOTensorName(0);
    const char *outputName = engine->getIOTensorName(1);

    // 获取张量形状
    nvinfer1::Dims inputDims = engine->getTensorShape(inputName);
    nvinfer1::Dims outputDims = engine->getTensorShape(outputName);

    // 打印输出张量形状用于调试
    std::cout << "Output tensor shape: [";
    for (int i = 0; i < outputDims.nbDims; ++i)
    {
        std::cout << outputDims.d[i];
        if (i < outputDims.nbDims - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 检测模型类型
    result.modelType = detectModelType(outputDims);
    switch (result.modelType)
    {
    case ModelType::YOLO11:
        result.modelTypeName = "YOLO11 (legacy)";
        break;
    case ModelType::YOLO26:
        result.modelTypeName = "YOLO26 (NMS-free)";
        break;
    default:
        result.modelTypeName = "Unknown (defaulting to YOLO11)";
        result.modelType = ModelType::YOLO11;
        break;
    }
    std::cout << "Detected model: " << result.modelTypeName << std::endl;

    auto calculateElementCount = [](const nvinfer1::Dims &dims)
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

    // 校验输入数据长度
    if (inputData.size() != inputElementCount)
    {
        throw std::runtime_error(
            "Input data size mismatch: expected " + std::to_string(inputElementCount) +
            ", got " + std::to_string(inputData.size()));
    }

    // 分配 GPU 内存，这里使用自定义的 CudaBuffer 类，会自动释放
    CudaBuffer<float> dInput(inputElementCount), dOutput(outputElementCount);
    if (dInput.empty() || dOutput.empty())
    {
        throw std::runtime_error("Failed to allocate device memory");
    }

    // 设置张量地址（TensorRT 10.x 推荐使用 setTensorAddress）
    context->setTensorAddress(inputName, dInput.get());
    context->setTensorAddress(outputName, dOutput.get());

    // 拷贝输入数据到 GPU
    dInput.copyFromHost(inputData.data(), inputElementCount);

    TEST_TIME_END(preInfer)

    TEST_TIME_BEGIN(infer)

    // 执行推理
    CudaStream stream;
    bool success = context->enqueueV3(stream.get());
    if (!success)
    {
        throw std::runtime_error("Inference enqueue failed");
    }

    // 等待流执行完成（确保推理结束后再拷贝数据）
    stream.synchronize();

    TEST_TIME_END(infer)

    UPDATE_TEST_TIME(postProcess)

    result.outputData.resize(outputElementCount);
    dOutput.copyToHost(result.outputData.data(), outputElementCount);

    return result;
}

// 主函数示例
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <path-to-engine-file>" << " <path-to-input-image>" << std::endl;
        std::cerr << "Supports both YOLO11 and YOLO26 models (auto-detected)" << std::endl;
        return 1;
    }

    try
    {
        TEST_TIME_BEGIN(loadImage)

        // 加载并预处理输入图片
        cv::Mat img = cv::imread(argv[2]);
        if (img.empty())
        {
            throw std::runtime_error("Failed to read input image: " + std::string(argv[2]));
        }

        TEST_TIME_END(loadImage)

        TEST_TIME_BEGIN(preprocessImage)
        std::vector<float> inputData = preprocessImage(img);
        TEST_TIME_END(preprocessImage)

        InferenceResult inferResult = runInference(argv[1], inputData);

        auto detectResults = postprocess(img, inferResult.outputData, inferResult.modelType);

        TEST_TIME_END(postProcess)

        std::cout << "\n=== Detection Results (" << detectResults.size() << " objects) ===" << std::endl;
        for (size_t i = 0; i < detectResults.size(); i++)
        {
            const auto &box = detectResults[i];
            std::cout << "[" << i + 1 << "] " << box.class_name << " (ID: " << box.class_id << ") "
                      << "Conf: " << box.conf << " "
                      << "Box: [" << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << "]"
                      << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
