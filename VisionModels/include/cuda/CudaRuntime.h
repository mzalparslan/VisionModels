#pragma once

#include <cstddef>
#include <string>

#include "Exceptions.h"

/**
 * @brief Interface to the optional CUDA (NVIDIA GPU) code.
 *
 * This header is plain C++: it includes nothing from the CUDA Toolkit, so any
 * project can include it and be compiled by MSVC or GCC alone. The
 * implementation lives in the VisionModels.Cuda project, which links into
 * every build:
 *  - built with the CUDA Toolkit (CudaRuntime.cu): the functions run on the GPU;
 *  - built without it (CudaUnavailable.cpp): isAvailable() is false and the
 *    other functions throw CudaError, so callers can fall back to the CPU.
 *
 * Always check isAvailable() before relying on the GPU.
 */
namespace cuda {

	/**
	 * @brief What is known about the GPU in use (device 0).
	 */
	class DeviceInfo {
	public:
		std::string name;
		// Total memory of the device in bytes.
		std::size_t totalMemoryBytes = 0;
		// Compute capability, e.g. 8.9 for an RTX 4060 Ti (Ada).
		int computeMajor = 0;
		int computeMinor = 0;
		// Number of streaming multiprocessors.
		int multiprocessorCount = 0;
	};

	/**
	 * @brief Whether this build can use CUDA and a CUDA device is present.
	 * Never throws.
	 */
	bool isAvailable();

	/**
	 * @brief Describes the GPU in use.
	 *
	 * @throws CudaError If CUDA is not available.
	 */
	DeviceInfo deviceInfo();

	/**
	 * @brief The GPU as one line of text, e.g.
	 * "NVIDIA GeForce RTX 4060 Ti, 8188 MiB, compute capability 8.9, 34 multiprocessors",
	 * or a note saying CUDA is unavailable. Never throws.
	 */
	inline std::string describeDevice() {
		if (!isAvailable()) {
			return "CUDA is not available";
		}
		const DeviceInfo info = deviceInfo();
		return info.name + ", " + std::to_string(info.totalMemoryBytes / (1024 * 1024)) + " MiB, compute capability "
			+ std::to_string(info.computeMajor) + "." + std::to_string(info.computeMinor) + ", "
			+ std::to_string(info.multiprocessorCount) + " multiprocessors";
	}

} // namespace cuda
