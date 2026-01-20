#pragma once

#include "NvInfer.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(status)                                         \
    do                                                             \
    {                                                              \
        auto ret = (status);                                       \
        if (ret != 0)                                              \
        {                                                          \
            std::cerr << "Cuda failure: " << ret << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            abort();                                               \
        }                                                          \
    } while (0)


class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8Calibrator(int batchSize, int inputW, int inputH, const std::string& calibDataPath, const std::string& calibCachePath, const std::string& inputBlobName)
        : mBatchSize(batchSize), mInputW(inputW), mInputH(inputH), mCalibCachePath(calibCachePath), mInputBlobName(inputBlobName)
    {
        mInputCount = mBatchSize * 3 * mInputW * mInputH;
        mDeviceInput = nullptr;
        
        // Find all image files in the calibration directory
        for (const auto& entry : std::filesystem::directory_iterator(calibDataPath)) {
            if (entry.is_regular_file()) {
                mImageFiles.push_back(entry.path().string());
            }
        }

        if (mImageFiles.empty()) {
            throw std::runtime_error("No calibration files found in: " + calibDataPath);
        }
        
        std::cout << "Found " << mImageFiles.size() << " calibration images." << std::endl;

        // Allocate GPU memory for the input batch
        CUDA_CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    }

    ~Int8Calibrator() override
    {
        if (mDeviceInput)
        {
            cudaFree(mDeviceInput);
            mDeviceInput = nullptr;
        }
    }

    int getBatchSize() const noexcept override
    {
        return mBatchSize;
    }

    // This function is called by TensorRT to get a batch of calibration data.
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        if (mImageIndex + mBatchSize > mImageFiles.size())
        {
            // All files have been processed
            return false;
        }

        // Preprocess a batch of images and copy to device
        std::vector<float> hostInput(mInputCount);
        const int imageSize = 3 * mInputW * mInputH;

        for (int i = 0; i < mBatchSize; ++i)
        {
            const std::string& imagePath = mImageFiles[mImageIndex + i];
            cv::Mat img = cv::imread(imagePath);
            if (img.empty()) {
                std::cerr << "Warning: Could not read image: " << imagePath << ". Skipping." << std::endl;
                continue;
            }

            // Preprocess the image (resize, BGR->RGB, HWC->CHW, normalize)
            cv::Mat resizedImg;
            cv::resize(img, resizedImg, cv::Size(mInputW, mInputH));
            
            cv::Mat rgbImg;
            cv::cvtColor(resizedImg, rgbImg, cv::COLOR_BGR2RGB);

            cv::Mat floatImg;
            rgbImg.convertTo(floatImg, CV_32F, 1.0 / 255.0);

            // HWC to CHW
            std::vector<cv::Mat> channels(3);
            cv::split(floatImg, channels);

            // Copy to the correct position in the batch buffer
            const int channelSize = mInputW * mInputH;
            memcpy(hostInput.data() + i * imageSize,                   channels[0].data, channelSize * sizeof(float));
            memcpy(hostInput.data() + i * imageSize + channelSize,     channels[1].data, channelSize * sizeof(float));
            memcpy(hostInput.data() + i * imageSize + 2 * channelSize, channels[2].data, channelSize * sizeof(float));
        }

        // Copy batch from host to device
        CUDA_CHECK(cudaMemcpy(mDeviceInput, hostInput.data(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        
        // Find the binding for the input blob
        int inputBindingIndex = -1;
        for (int i = 0; i < nbBindings; ++i) {
            if (strcmp(names[i], mInputBlobName.c_str()) == 0) {
                inputBindingIndex = i;
                break;
            }
        }
        if (inputBindingIndex == -1) {
             std::cerr << "Error: Could not find input blob with name '" << mInputBlobName << "' in network bindings." << std::endl;
             return false;
        }

        bindings[inputBindingIndex] = mDeviceInput;
        mImageIndex += mBatchSize;
        std::cout << "Calibrating with batch " << (mImageIndex / mBatchSize) << "/" << (mImageFiles.size() / mBatchSize) << std::endl;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibCachePath, std::ios::binary);
        if (input.good())
        {
            input.seekg(0, std::ios::end);
            length = input.tellg();
            input.seekg(0, std::ios::beg);
            mCalibrationCache.resize(length);
            input.read(mCalibrationCache.data(), length);
            input.close();
             std::cout << "Using existing calibration cache: " << mCalibCachePath << std::endl;
        } else {
            length = 0;
        }
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        std::cout << "Writing calibration cache to: " << mCalibCachePath << " (" << length << " bytes)" << std::endl;
        std::ofstream output(mCalibCachePath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
        output.close();
    }

private:
    int mBatchSize;
    int mInputW;
    int mInputH;
    size_t mInputCount;
    size_t mImageIndex{0};

    std::string mCalibCachePath;
    std::string mInputBlobName;

    std::vector<std::string> mImageFiles;
    void* mDeviceInput;
    std::vector<char> mCalibrationCache;
};
