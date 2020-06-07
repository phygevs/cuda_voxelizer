#include "util_cuda.h"

// Check if CUDA requirements are met
bool initCuda()
{
    logging::logger_t & logger_main = logging::logger_main::get();

    int device_count{};
    // Check if CUDA runtime calls work at all
    cudaError t = cudaGetDeviceCount(&device_count);
    if (t != cudaSuccess) {
        BOOST_LOG_SEV(logger_main, logging::severity_t::error) << "[CUDA] First call to CUDA Runtime API failed. Are the drivers installed?" << std::endl;
        return false;
    }

    // Is there a CUDA device at all?
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count < 1) {
        BOOST_LOG_SEV(logger_main, logging::severity_t::error) << "[CUDA] No CUDA devices found\n" <<
            "[CUDA] Make sure CUDA device is powered, connected and available.\n" <<
            "[CUDA] On laptops: disable powersave/battery mode.\n" <<
            "[CUDA] Exiting..." << std::endl;
        return false;
    }

    BOOST_LOG_SEV(logger_main, logging::severity_t::trace) << "[CUDA] CUDA device(s) found, picking best one\n" <<
        "[CUDA] ";
    // We have at least 1 CUDA device, so now select the fastest (method from Nvidia helper library)
    int device = findCudaDevice(0, 0);

    // Print available device memory
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, device));
    BOOST_LOG_SEV(logger_main, logging::severity_t::trace) << "[CUDA] Best device: " << properties.name << std::endl;
    BOOST_LOG_SEV(logger_main, logging::severity_t::trace) << "[CUDA] Available global device memory: " << (properties.totalGlobalMem / (1024.0 * 1024)) << " MB." << std::endl;

    // Check compute capability
    if (properties.major < 2)
    {
        BOOST_LOG_SEV(logger_main, logging::severity_t::trace) << "[CUDA] Your cuda device has compute capability " << properties.major << '.' << properties.minor << ". We need at least 2.0 for atomic operations. Exiting." << std::endl;
        return false;
    }

    return true;
}