#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "common.h"

// Preprocess image for YOLO inference
// Returns CHW format normalized float vector
inline std::vector<float> preprocessImage(const cv::Mat& img, const YoloParams& params)
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
