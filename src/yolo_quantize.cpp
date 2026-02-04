#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "trt_logger.h"
#include "int8_calibrator.h"

// Configuration structure
struct QuantizeConfig
{
    std::string onnxFile;
    std::string engineFile;
    std::string calibDataPath = "./calibration_data";
    std::string calibCacheFile = "yolo.cache";
    int calibrationBatchSize = 8;
    int inputW = 640;
    int inputH = 640;
    std::string inputBlobName = "images";
    size_t maxWorkspaceSize = 1U << 30;  // 1 GB
};

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " <onnx_file> <output_engine_file> [options]\n\n";
    std::cout << "Required arguments:\n";
    std::cout << "  onnx_file           Path to input ONNX model file\n";
    std::cout << "  output_engine_file  Path for output TensorRT engine file\n\n";
    std::cout << "Optional arguments:\n";
    std::cout << "  --calib-path PATH       Path to calibration images directory (default: ./calibration_data)\n";
    std::cout << "  --cache-file FILE       Calibration cache file name (default: yolo.cache)\n";
    std::cout << "  --batch-size N          Calibration batch size (default: 8)\n";
    std::cout << "  --input-width W         Input image width (default: 640)\n";
    std::cout << "  --input-height H        Input image height (default: 640)\n";
    std::cout << "  --input-name NAME       Input tensor name in ONNX model (default: images)\n";
    std::cout << "  --workspace SIZE        Max workspace size in MB (default: 1024)\n";
    std::cout << "  --help                  Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << programName << " yolo.onnx yolo_int8.engine --calib-path ./calib_images --batch-size 4\n";
}

void buildEngine(const QuantizeConfig& config, TrtLogger& logger)
{
    // 1. Create Builder
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder.");
    }

    // 2. Create Network Definition
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
    std::cout << "Parsing ONNX model: " << config.onnxFile << std::endl;
    std::ifstream onnxFile(config.onnxFile, std::ios::binary);
    if (!onnxFile.good()) {
        throw std::runtime_error("Could not open ONNX file: " + config.onnxFile);
    }

    onnxFile.seekg(0, std::ios::end);
    size_t size = onnxFile.tellg();
    onnxFile.seekg(0, std::ios::beg);
    std::vector<char> onnxModel(size);
    onnxFile.read(onnxModel.data(), size);

    if (!parser->parse(onnxModel.data(), size)) {
        std::string errorMsg = "Failed to parse the ONNX file.";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            errorMsg += "\n" + std::string(parser->getError(i)->desc());
        }
        throw std::runtime_error(errorMsg);
    }
    std::cout << "Successfully parsed ONNX model." << std::endl;

    // 5. Create Builder Config
    std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig(builder->createBuilderConfig());
    if (!builderConfig) {
        throw std::runtime_error("Failed to create builder config.");
    }

    // Set max workspace size
    builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.maxWorkspaceSize);
    std::cout << "Max workspace size: " << (config.maxWorkspaceSize >> 20) << " MB" << std::endl;

    // 6. Setup for INT8 Quantization
    std::cout << "Setting up for INT8 calibration..." << std::endl;
    builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);

    auto calibrator = std::make_unique<Int8Calibrator>(
        config.calibrationBatchSize, config.inputW, config.inputH,
        config.calibDataPath, config.calibCacheFile, config.inputBlobName);

    builderConfig->setInt8Calibrator(calibrator.get());

    // 7. Build the Engine
    std::cout << "Building the INT8 engine. This may take a while..." << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *builderConfig));
    if (!serializedEngine) {
        throw std::runtime_error("Failed to build the engine.");
    }
    std::cout << "Engine built successfully." << std::endl;

    // 8. Save the Engine to a File
    std::cout << "Saving engine to file: " << config.engineFile << std::endl;
    std::ofstream engineFile(config.engineFile, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    std::cout << "INT8 engine saved to " << config.engineFile << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    QuantizeConfig config;
    config.onnxFile = argv[1];
    config.engineFile = argv[2];

    // Parse optional arguments
    for (int i = 3; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--calib-path" && i + 1 < argc)
        {
            config.calibDataPath = argv[++i];
        }
        else if (arg == "--cache-file" && i + 1 < argc)
        {
            config.calibCacheFile = argv[++i];
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            config.calibrationBatchSize = std::stoi(argv[++i]);
        }
        else if (arg == "--input-width" && i + 1 < argc)
        {
            config.inputW = std::stoi(argv[++i]);
        }
        else if (arg == "--input-height" && i + 1 < argc)
        {
            config.inputH = std::stoi(argv[++i]);
        }
        else if (arg == "--input-name" && i + 1 < argc)
        {
            config.inputBlobName = argv[++i];
        }
        else if (arg == "--workspace" && i + 1 < argc)
        {
            size_t workspaceMB = std::stoull(argv[++i]);
            config.maxWorkspaceSize = workspaceMB << 20;  // Convert MB to bytes
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Print configuration
    std::cout << "\n=== Quantization Configuration ===" << std::endl;
    std::cout << "ONNX file:         " << config.onnxFile << std::endl;
    std::cout << "Engine file:       " << config.engineFile << std::endl;
    std::cout << "Calib data path:   " << config.calibDataPath << std::endl;
    std::cout << "Cache file:        " << config.calibCacheFile << std::endl;
    std::cout << "Batch size:        " << config.calibrationBatchSize << std::endl;
    std::cout << "Input size:        " << config.inputW << "x" << config.inputH << std::endl;
    std::cout << "Input blob name:   " << config.inputBlobName << std::endl;
    std::cout << "Max workspace:     " << (config.maxWorkspaceSize >> 20) << " MB" << std::endl;
    std::cout << "==================================\n" << std::endl;

    // Logger is required for the builder, network, and parser
    TrtLogger logger(nvinfer1::ILogger::Severity::kINFO);

    try {
        buildEngine(config, logger);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
