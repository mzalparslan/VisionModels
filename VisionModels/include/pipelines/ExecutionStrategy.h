#pragma once

#include <string>

#include "Exceptions.h"

/**
 * @brief How a ConvNet is trained and evaluated.
 */
enum class ExecutionStrategy {
	// One thread. Every layer runs its loops in order on the calling thread.
	Sequential,

	// Several CPU threads. Each layer splits its work into independent pieces
	// (one output feature map, one sample, one filter's gradient) and runs them
	// on ThreadPool::shared(). Every number is summed in the same order as in
	// Sequential, so the results are bit for bit the same, only faster.
	Parallel,

	// The NVIDIA GPU (cuda::GpuConvNet). Weights, gradients, optimizer state and
	// the training images stay in GPU memory for the whole run; only batch
	// indices go up and the loss comes back each step. float only. Results match
	// Sequential to within rounding, not bit for bit (the GPU sums in a
	// different order and fuses multiply-adds). Needs a CUDA device; training
	// with it throws CudaError otherwise.
	Cuda
};

/**
 * @brief Name of a strategy, for logging.
 */
inline const char* toString(ExecutionStrategy strategy) {
	switch (strategy) {
	case ExecutionStrategy::Sequential:
		return "Sequential";
	case ExecutionStrategy::Parallel:
		return "Parallel";
	case ExecutionStrategy::Cuda:
		return "Cuda";
	}
	return "Unknown";
}

/**
 * @brief Parses "sequential", "parallel" or "cuda" (lower case).
 *
 * @throws InvalidParameterError For any other text.
 */
inline ExecutionStrategy parseExecutionStrategy(const std::string& text) {
	if (text == "sequential") {
		return ExecutionStrategy::Sequential;
	}
	if (text == "parallel") {
		return ExecutionStrategy::Parallel;
	}
	if (text == "cuda") {
		return ExecutionStrategy::Cuda;
	}
	throw InvalidParameterError("Unknown execution strategy '" + text + "' (use sequential, parallel or cuda)!");
}
