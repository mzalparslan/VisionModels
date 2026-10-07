#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Layer.h"

/**
 * @brief Turns each sample's feature maps into one feature vector:
 * [N, C, H, W] to [N, C * H * W]. The bridge from convolution layers to
 * Dense layers.
 *
 * NCHW data is already stored sample by sample, so this only changes the
 * shape; backward changes it back.
 */
template <typename T>
class Flatten : public Layer<T> {
public:
	LayerSpec spec() const override {
		return LayerSpec::flatten();
	}

	std::string describe() const override {
		return "Flatten";
	}

	std::vector<std::size_t> outputShape(const std::vector<std::size_t>& inputShape) const override {
		if (inputShape.empty()) {
			throw InvalidSizeError("Flatten needs an input with at least one dimension!");
		}
		return { Tensor<T>::elementCount(inputShape) };
	}

	Tensor<T> forward(const Tensor<T>& input, std::size_t /*threads*/) override {
		if (input.rank() < 2 || input.shape[0] == 0) {
			throw InvalidSizeError("Flatten needs a non-empty batch input!");
		}
		cachedInputShape = input.shape;
		Tensor<T> output = input;
		output.reshape({ input.shape[0], input.size() / input.shape[0] });
		return output;
	}

	Tensor<T> backward(const Tensor<T>& gradOutput, std::size_t /*threads*/) override {
		if (cachedInputShape.empty()) {
			throw PipelineStateError("Flatten: backward() called before forward()!");
		}
		Tensor<T> gradInput = gradOutput;
		gradInput.reshape(cachedInputShape);
		return gradInput;
	}

private:
	std::vector<std::size_t> cachedInputShape;
};
