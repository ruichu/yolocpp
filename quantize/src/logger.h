#pragma once

#include "NvInfer.h"
#include <iostream>

// A simple logger class for TensorRT messages.
class Logger : public nvinfer1::ILogger
{
public:
    explicit Logger(Severity severity = Severity::kWARNING) : m_severity(severity) {}

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
