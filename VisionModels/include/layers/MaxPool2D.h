#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Layer.h"
#include "ParallelFor.h"
#include "Validation.h"

/**
 * @brief 2-D max pooling: keeps the largest value of each K x K window,
 * channel by channel.
 *
 * Input [N, C, H, W], output [N, C, OH, OW] with OH = (H - K) / stride + 1.
 * The usual 2x2 window with stride 2 halves the width and height, so later
 * layers see a larger part of the image for the same cost, and a feature that
 * moves by a pixel often still wins the same window (a little translation
 * invariance). No weights.
 *
 * Backward: only the winner of each window influenced the output, so the
 * whole gradient goes to it and the other positions get 0. forward() records
 * the winner of every window (the first, if several tie) for that.
 */
template <typename T>
class MaxPool2D : public Layer<T> {
public:
	/**
	 * @throws InvalidParameterSizeError If kernelSize or stride is 0.
	 */
	MaxPool2D(std::size_t kernelSize, std::size_t stride)
		: kernelSize(kernelSize), stride(stride)
	{
		validation::requirePositiveSize(kernelSize, "MaxPool2D kernel size");
		validation::requirePositiveSize(stride, "MaxPool2D stride");
	}

	LayerSpec spec() const override {
		return LayerSpec::maxPool2D(kernelSize, stride);
	}

	std::string describe() const override {
		return "MaxPool2D(" + std::to_string(kernelSize) + "x" + std::to_string(kernelSize)
			+ ", stride " + std::to_string(stride) + ")";
	}

	std::vector<std::size_t> outputShape(const std::vector<std::size_t>& inputShape) const override {
		if (inputShape.size() != 3) {
			throw InvalidSizeError(describe() + " needs a [C, H, W] input!");
		}
		const std::size_t outHeight = slidingWindowOutputSize(inputShape[1], kernelSize, stride, 0);
		const std::size_t outWidth = slidingWindowOutputSize(inputShape[2], kernelSize, stride, 0);
		if (outHeight == 0 || outWidth == 0) {
			throw InvalidSizeError(describe() + ": the input is smaller than the window!");
		}
		return { inputShape[0], outHeight, outWidth };
	}

	Tensor<T> forward(const Tensor<T>& input, std::size_t threads) override {
		if (input.rank() != 4 || input.shape[0] == 0) {
			throw InvalidSizeError(describe() + " needs a non-empty [N, C, H, W] input!");
		}
		const std::vector<std::size_t> sampleOut = outputShape({ input.shape[1], input.shape[2], input.shape[3] });

		const std::size_t planes = input.shape[0] * input.shape[1];
		const std::size_t height = input.shape[2];
		const std::size_t width = input.shape[3];
		const std::size_t outHeight = sampleOut[1];
		const std::size_t outWidth = sampleOut[2];

		Tensor<T> output({ input.shape[0], input.shape[1], outHeight, outWidth });
		winners.assign(output.size(), 0);

		// One item = one (n, c) plane; windows never cross planes.
		parallelFor(planes, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t plane = begin; plane < end; plane++) {
				const std::size_t planeStart = plane * height * width;
				for (std::size_t oh = 0; oh < outHeight; oh++) {
					for (std::size_t ow = 0; ow < outWidth; ow++) {
						std::size_t best = planeStart + (oh * stride) * width + ow * stride;
						for (std::size_t kh = 0; kh < kernelSize; kh++) {
							for (std::size_t kw = 0; kw < kernelSize; kw++) {
								const std::size_t index = planeStart + (oh * stride + kh) * width + ow * stride + kw;
								if (input[index] > input[best]) {
									best = index;
								}
							}
						}
						const std::size_t outIndex = (plane * outHeight + oh) * outWidth + ow;
						output[outIndex] = input[best];
						winners[outIndex] = best;
					}
				}
			}
		});

		cachedInputShape = input.shape;
		cachedOutputShape = output.shape;
		return output;
	}

	Tensor<T> backward(const Tensor<T>& gradOutput, std::size_t threads) override {
		if (cachedOutputShape.empty()) {
			throw PipelineStateError(describe() + ": backward() called before forward()!");
		}
		if (gradOutput.shape != cachedOutputShape) {
			throw InvalidSizeError(describe() + ": gradient shape differs from the forward output!");
		}

		Tensor<T> gradInput(cachedInputShape, T(0));
		const std::size_t planes = cachedOutputShape[0] * cachedOutputShape[1];
		const std::size_t outPlaneSize = cachedOutputShape[2] * cachedOutputShape[3];

		// Windows overlap when stride < kernel, so one input can win several
		// windows; each plane is still handled by one thread, in order.
		parallelFor(planes, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t outIndex = begin * outPlaneSize; outIndex < end * outPlaneSize; outIndex++) {
				gradInput[winners[outIndex]] += gradOutput[outIndex];
			}
		});
		return gradInput;
	}

private:
	std::size_t kernelSize;
	std::size_t stride;

	// Flat input index of the winner of every output element.
	std::vector<std::size_t> winners;
	std::vector<std::size_t> cachedInputShape;
	std::vector<std::size_t> cachedOutputShape;
};
