#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Common YOLO inference parameters
struct YoloParams
{
    int inputHeight = 640;
    int inputWidth = 640;
    float confThresh = 0.5f;
    float nmsThresh = 0.5f;
};

// Model type enumeration
enum class ModelType
{
    UNKNOWN,
    YOLO_V8,  // Traditional format: output (1, 84, 8400) or similar, requires NMS
    YOLO_V26  // New format: output (1, 300, 6), NMS-free
};

// Detection box structure
struct DetectBox
{
    float x1, y1, x2, y2;
    float conf;
    int class_id;
    std::string class_name;
};

// COCO class names (80 classes)
const std::vector<std::string> COCO_CLASS_NAMES =
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

// Get class name safely
inline std::string getClassName(int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(COCO_CLASS_NAMES.size()))
    {
        return COCO_CLASS_NAMES[class_id];
    }
    return "unknown";
}
