#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cuda_runtime.h>

// CUDA错误检查宏（封装核心）
#define CHECK_CUDA_ERROR(call)                                                                         \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) +           \
                                     " at line " + std::to_string(__LINE__) + " in file " + __FILE__); \
        }                                                                                              \
    } while (0)

// CUDA设备内存管理类（RAII）
template <typename T>
class CudaBuffer
{
protected:
    // 手动分配内存
    void allocate(size_t count, int device_id = 0)
    {
        if (data_)
        {
            free(); // 释放已有内存
        }
        size_ = count;
        device_id_ = device_id;

        // 切换到目标设备
        CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
        // 分配设备内存
        CHECK_CUDA_ERROR(cudaMalloc(&data_, count * sizeof(T)));
    }

    // 手动释放内存
    void free()
    {
        if (data_)
        {
            // 切换到内存所在设备
            CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
            CHECK_CUDA_ERROR(cudaFree(data_));
            data_ = nullptr;
            size_ = 0;
        }
    }

public:
    // 空构造函数
    CudaBuffer() : data_(nullptr), size_(0), device_id_(0) {}

    // 带大小的构造函数（自动分配内存）
    explicit CudaBuffer(size_t count, int device_id = 0)
        : size_(count), device_id_(device_id)
    {
        allocate(count, device_id);
    }

    // 禁止拷贝（避免浅拷贝导致重复释放）
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    // 移动构造（支持返回值优化）
    CudaBuffer(CudaBuffer &&other) noexcept
    {
        data_ = other.data_;
        size_ = other.size_;
        device_id_ = other.device_id_;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 移动赋值
    CudaBuffer &operator=(CudaBuffer &&other) noexcept
    {
        if (this != &other)
        {
            free(); // 释放当前内存
            data_ = other.data_;
            size_ = other.size_;
            device_id_ = other.device_id_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // 析构函数（自动释放）
    ~CudaBuffer()
    {
        free();
    }

    // 获取设备指针
    T *get() const
    {
        return data_;
    }

    // 获取元素数量
    size_t size() const
    {
        return size_;
    }

    // 获取字节数
    size_t bytes() const
    {
        return size_ * sizeof(T);
    }

    // 判断是否为空
    bool empty() const
    {
        return data_ == nullptr || size_ == 0;
    }

    // 内存拷贝：主机 → 设备（同步）
    void copyFromHost(const T *host_data, size_t count = 0)
    {
        if (host_data == nullptr)
        {
            throw std::invalid_argument("Host data pointer is null");
        }
        size_t copy_size = (count == 0) ? size_ : count;
        if (copy_size > size_)
        {
            throw std::out_of_range("Copy size exceeds buffer size");
        }

        CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
        CHECK_CUDA_ERROR(cudaMemcpy(data_, host_data, copy_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    // 内存拷贝：设备 → 主机（同步）
    void copyToHost(T *host_data, size_t count = 0) const
    {
        if (host_data == nullptr)
        {
            throw std::invalid_argument("Host data pointer is null");
        }
        size_t copy_size = (count == 0) ? size_ : count;
        if (copy_size > size_)
        {
            throw std::out_of_range("Copy size exceeds buffer size");
        }

        CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
        CHECK_CUDA_ERROR(cudaMemcpy(host_data, data_, copy_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // 内存拷贝：设备 → 设备（同步）
    void copyFromDevice(const CudaBuffer<T> &other)
    {
        if (other.empty())
        {
            throw std::invalid_argument("Source buffer is empty");
        }
        if (size_ < other.size_)
        {
            throw std::out_of_range("Destination buffer is smaller than source");
        }

        CHECK_CUDA_ERROR(cudaMemcpyPeer(data_, device_id_, other.data_, other.device_id_,
                                        other.size_ * sizeof(T)));
    }

private:
    T *data_ = nullptr; // 设备内存指针
    size_t size_ = 0;   // 元素数量
    int device_id_ = 0; // 内存所在设备ID
};

// CUDA流封装类
class CudaStream
{
public:
    CudaStream(int device_id = 0) : device_id_(device_id)
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
    }

    ~CudaStream()
    {
        if (stream_)
        {
            try
            {
                CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
                CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
            }

            stream_ = nullptr;
        }
    }

    // 禁止拷贝
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    // 移动语义
    CudaStream(CudaStream &&other) noexcept
    {
        stream_ = other.stream_;
        device_id_ = other.device_id_;
        other.stream_ = nullptr;
    }

    CudaStream &operator=(CudaStream &&other) noexcept
    {
        if (this != &other)
        {
            if (stream_)
            {
                try
                {
                    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
                }
                catch (const std::exception &e)
                {
                    std::cerr << e.what() << '\n';
                }
            }
            stream_ = other.stream_;
            device_id_ = other.device_id_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    // 获取流句柄
    cudaStream_t get() const
    {
        return stream_;
    }

    // 等待流完成
    void synchronize() const
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
    }

    // 检查流是否完成
    bool isFinished() const
    {
        CHECK_CUDA_ERROR(cudaSetDevice(device_id_));
        return cudaStreamQuery(stream_) == cudaSuccess;
    }

private:
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
};

// 辅助函数：获取当前设备ID
inline int getCurrentCudaDevice()
{
    int dev_id;
    CHECK_CUDA_ERROR(cudaGetDevice(&dev_id));
    return dev_id;
}

// 辅助函数：设置当前设备
inline void setCurrentCudaDevice(int dev_id)
{
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id));
}

// 辅助函数：获取设备数量
inline int getCudaDeviceCount()
{
    int count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&count));
    return count;
}

// 测试示例
int test()
{
    try
    {
        // 1. 基本内存管理
        const size_t n_elems = 1024;
        CudaBuffer<int> cuda_buf(n_elems); // 自动分配内存

        // 主机数据
        int *host_data = new int[n_elems];
        for (size_t i = 0; i < n_elems; ++i)
        {
            host_data[i] = static_cast<int>(i);
        }

        // 2. 主机 → 设备拷贝
        cuda_buf.copyFromHost(host_data);

        // 3. 设备 → 主机拷贝
        int *host_out = new int[n_elems];
        cuda_buf.copyToHost(host_out);

        // 验证数据
        for (size_t i = 0; i < n_elems; ++i)
        {
            assert(host_out[i] == host_data[i]);
        }

        // 4. 多设备示例（如果有多个GPU）
        int dev_count = getCudaDeviceCount();
        if (dev_count > 1)
        {
            CudaBuffer<int> cuda_buf2(n_elems, 1); // 在设备1上分配
            cuda_buf.copyFromDevice(cuda_buf2);    // 设备间拷贝
        }

        // 5. 流使用示例
        CudaStream stream;
        // 核函数异步执行（示例）
        // kernel<<<grid, block, 0, stream.get()>>>(cuda_buf.get(), n_elems);
        stream.synchronize();

        // 释放主机内存
        delete[] host_data;
        delete[] host_out;

        std::cout << "All operations completed successfully!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}