// The implementation of CudaRuntime.h used when the build has no CUDA Toolkit
// (no nvcc, or a platform CUDA does not support, such as Win32). It lets every
// project link and run: isAvailable() is false, and callers fall back to the CPU.

#include "CudaRuntime.h"

namespace cuda {

	bool isAvailable() {
		return false;
	}

	DeviceInfo deviceInfo() {
		throw CudaError("cuda::deviceInfo: This build has no CUDA support.");
	}

} // namespace cuda
