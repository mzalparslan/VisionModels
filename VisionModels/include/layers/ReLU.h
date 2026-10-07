#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Layer.h"
#include "ParallelFor.h"

/**
 * @brief Rectified Linear Unit, element by element: out = max(0, in).
 *
 * The non-linearity between layers: without it a stack of convolutions and
 * dense layers would collapse into one linear map. Its gradient is 1 where the
 * input was positive and 0 elsewhere, so it passes gradients through unchanged
 * or blocks them, and does not shrink them the way sigmoid does.
 *
 * Works on any shape.
 */
template <typename T>
class ReLU : public Layer<T> {
public:
	LayerSpec spec() const override {
		return LayerSpec::relu();
	}

	std::string describe() const override {
		return "ReLU";
	}

	std::vector<std::size_t> outputShape(const std::vector<std::size_t>& inputShape) const override {
		return inputShape;
	}

	Tensor<T> forward(const Tensor<T>& input, std::size_t threads) override {
		if (input.size() == 0) {
			throw InvalidSizeError("ReLU needs a non-empty input!");
		}
		cachedInput = input;
		Tensor<T> output(input.shape);
		parallelFor(input.size(), minElementsPerThread, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t i = begin; i < end; i++) {
				output[i] = input[i] > T(0) ? input[i] : T(0);
			}
		});
		return output;
	}

	Tensor<T> backward(const Tensor<T>& gradOutput, std::size_t threads) override {
		if (cachedInput.shape.empty()) {
			throw PipelineStateError("ReLU: backward() called before forward()!");
		}
		if (gradOutput.shape != cachedInput.shape) {
			throw InvalidSizeError("ReLU: gradient shape differs from the forward output!");
		}
		Tensor<T> gradInput(gradOutput.shape);
		parallelFor(gradOutput.size(), minElementsPerThread, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t i = begin; i < end; i++) {
				gradInput[i] = cachedInput[i] > T(0) ? gradOutput[i] : T(0);
			}
		});
		return gradInput;
	}

private:
	// Below this, waking a thread costs more than the work it would do.
	static constexpr std::size_t minElementsPerThread = 16384;

	Tensor<T> cachedInput;
};
