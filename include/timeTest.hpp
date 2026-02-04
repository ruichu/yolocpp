#include <chrono>  // C++ 标准计时库
#include <iostream>
#include <iomanip>

//std::chrono::high_resolution_clock

#define TEST_TIME_BEGIN(name) \
    auto name##_start = std::chrono::high_resolution_clock::now();

#define UPDATE_TEST_TIME(name) \
    name##_start = std::chrono::high_resolution_clock::now();

#define TEST_TIME_END(name) \
    auto name##_end = std::chrono::high_resolution_clock::now(); \
    auto name##_duration = std::chrono::duration_cast<std::chrono::milliseconds>(name##_end - name##_start).count(); \
    std::cout << "Elapsed time for " #name ": " << name##_duration << " ms" << std::endl;
