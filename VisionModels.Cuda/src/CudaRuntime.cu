// The GPU implementation of CudaRuntime.h. Compiled by nvcc (see nvcc-build.cmd);
// only built when the CUDA Toolkit is installed.

#include "CudaRuntime.h"
#include "detail/CudaHost.cuh"

#include <cuda_runtime.h>

namespace cuda {

	bool isAvailable() {
		int count = 0;
		if (cudaGetDeviceCount(&count) != cudaSuccess) {
			// Clear the error state, so it is not reported by a later, unrelated call.
			(void)cudaGetLastError();
			return false;
		}
		return count > 0;
	}

	DeviceInfo deviceInfo() {
		if (!isAvailable()) {
			throw CudaError("cuda::deviceInfo: No CUDA device is available.");
		}

		cudaDeviceProp properties;
		detail::check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");

		DeviceInfo info;
		info.name = properties.name;
		info.totalMemoryBytes = properties.totalGlobalMem;
		info.computeMajor = properties.major;
		info.computeMinor = properties.minor;
		info.multiprocessorCount = properties.multiProcessorCount;
		return info;
	}

} // namespace cuda
