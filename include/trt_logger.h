#pragma once

#include <NvInfer.h>
#include <iostream>
#include <memory>

// A simple logger class for TensorRT messages
class TrtLogger : public nvinfer1::ILogger
{
public:
    explicit TrtLogger(Severity severity = Severity::kWARNING) : m_severity(severity) {}

    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= m_severity)
        {
            std::cerr << "[TRT] " << getSeverityName(severity) << ": " << msg << std::endl;
        }
    }

private:
    Severity m_severity;

    const char* getSeverityName(Severity severity)
    {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            case Severity::kERROR:   return "ERROR";
            case Severity::kWARNING: return "WARNING";
            case Severity::kINFO:    return "INFO";
            case Severity::kVERBOSE: return "VERBOSE";
            default: return "UNKNOWN";
        }
    }
};

// Smart pointer deleter for TensorRT objects (TensorRT 10.x compatible)
template <typename T>
struct TrtDeleter
{
    void operator()(T* ptr) const
    {
        if (ptr)
        {
            delete ptr;
        }
    }
};
