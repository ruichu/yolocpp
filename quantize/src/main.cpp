#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "logger.h"
#include "calibrator.h"

// --- Configuration ---
const std::string ONNX_FILE = "yolo11m.onnx";
const std::string ENGINE_FILE = "yolo11m_int8.engine";
const std::string CALIB_DATA_PATH = "./calibration_data";
const std::string CALIB_CACHE_FILE = "yolo11m.cache";
const int CALIBRATION_BATCH_SIZE = 8;
const int INPUT_W = 640;
const int INPUT_H = 640;
// The name of the input tensor in your ONNX model.
// You may need to change this. Use a tool like Netron to find the correct name.
const std::string INPUT_BLOB_NAME = "images"; 
// --- End Configuration ---

void buildEngine(Logger& logger)
{
    // 1. Create Builder
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder.");
    }

    // 2. Create Network Definition
    // The kEXPLICIT_BATCH flag is required for ONNX parsing
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    if (!network) {
        throw std::runtime_error("Failed to create TensorRT network definition.");
    }

    // 3. Create ONNX Parser
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        throw std::runtime_error("Failed to create ONNX parser.");
    }

    // 4. Parse ONNX Model
    std::cout << "Parsing ONNX model: " << ONNX_FILE << std::endl;
    std::ifstream onnxFile(ONNX_FILE, std::ios::binary);
    if (!onnxFile.good()) {
        throw std::runtime_error("Could not open ONNX file: " + ONNX_FILE);
    }
    
    onnxFile.seekg(0, std::ios::end);
    size_t size = onnxFile.tellg();
    onnxFile.seekg(0, std::ios::beg);
    std::vector<char> onnxModel(size);
    onnxFile.read(onnxModel.data(), size);
    
    if (!parser->parse(onnxModel.data(), size)) {
        throw std::runtime_error("Failed to parse the ONNX file.");
    }
    std::cout << "Successfully parsed ONNX model." << std::endl;

    // 5. Create Builder Config
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("Failed to create builder config.");
    }

    // Set max workspace size
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1 GB

    // 6. Setup for INT8 Quantization
    std::cout << "Setting up for INT8 calibration..." << std::endl;
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    
    auto calibrator = std::make_unique<Int8Calibrator>(
        CALIBRATION_BATCH_SIZE, INPUT_W, INPUT_H, CALIB_DATA_PATH, CALIB_CACHE_FILE, INPUT_BLOB_NAME);
    
    config->setInt8Calibrator(calibrator.get());

    // 7. Build the Engine
    std::cout << "Building the INT8 engine. This may take a while..." << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        throw std::runtime_error("Failed to build the engine.");
    }
    std::cout << "Engine built successfully." << std::endl;

    // 8. Save the Engine to a File
    std::cout << "Saving engine to file: " << ENGINE_FILE << std::endl;
    std::ofstream engineFile(ENGINE_FILE, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    std::cout << "INT8 engine saved to " << ENGINE_FILE << std::endl;
}

int main(int argc, char** argv)
{
    // A logger is required for the builder, network, and parser
    Logger logger(nvinfer1::ILogger::Severity::kINFO);

    try {
        buildEngine(logger);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
