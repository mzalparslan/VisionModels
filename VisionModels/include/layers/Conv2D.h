#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Layer.h"
#include "ParallelFor.h"
#include "Validation.h"

/**
 * @brief 2-D convolution: a bank of small learned filters slid over the image.
 *
 * Input [N, IC, H, W], output [N, OC, OH, OW] with
 * OH = (H + 2 * padding - K) / stride + 1 (the same for OW).
 *
 * Every output channel oc has one filter weights[oc] of shape [IC, K, K] and
 * one bias[oc]. At each position the filter is laid over a K x K window of
 * all IC input channels, and the output is the weighted sum plus the bias:
 *
 *   out[n, oc, oh, ow] = bias[oc]
 *       + sum over ic, kh, kw of weights[oc, ic, kh, kw] * in[n, ic, oh*S + kh - P, ow*S + kw - P]
 *
 * (input positions outside the image read as 0: that is the padding).
 *
 * Why it works for images: the same filter is used at every position (weight
 * sharing), so an edge detector learned in one corner works everywhere, and
 * the layer has IC*K*K*OC + OC weights however big the image is. A Dense layer
 * over a 28x28 image would need 784 weights per output.
 *
 * The bias: one per filter, shared by every position of its feature map. It
 * sets how strongly the filter's pattern must match before the next ReLU
 * lets the value through.
 *
 * This is the direct ("naive") algorithm, written to be read. Libraries use
 * im2col + matrix multiply, Winograd or FFT for speed.
 */
template <typename T>
class Conv2D : public Layer<T> {
public:
	// [OC, IC, K, K]
	Parameter<T> weights;
	// [OC]
	Parameter<T> bias;

	/**
	 * Weights start from He initialization, N(0, 2 / fanIn) with fanIn = IC*K*K,
	 * which keeps the size of activations steady through ReLU layers; biases start at 0.
	 *
	 * @throws InvalidParameterSizeError If a channel count, kernel size or stride is 0.
	 */
	Conv2D(std::size_t inChannels, std::size_t outChannels, std::size_t kernelSize,
		std::size_t stride, std::size_t padding, RandomEngine& rng)
		: inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize),
		stride(stride), padding(padding)
	{
		validation::requirePositiveSize(inChannels, "Conv2D input channels");
		validation::requirePositiveSize(outChannels, "Conv2D output channels");
		validation::requirePositiveSize(kernelSize, "Conv2D kernel size");
		validation::requirePositiveSize(stride, "Conv2D stride");

		const T fanIn = static_cast<T>(inChannels * kernelSize * kernelSize);
		weights.initNormal({ outChannels, inChannels, kernelSize, kernelSize }, std::sqrt(T(2) / fanIn), rng);
		bias.initConstant({ outChannels }, T(0));
	}

	LayerSpec spec() const override {
		return LayerSpec::conv2D(inChannels, outChannels, kernelSize, stride, padding);
	}

	std::string describe() const override {
		return "Conv2D(" + std::to_string(inChannels) + " -> " + std::to_string(outChannels) + ", "
			+ std::to_string(kernelSize) + "x" + std::to_string(kernelSize) + ", stride "
			+ std::to_string(stride) + ", padding " + std::to_string(padding) + ")";
	}

	std::vector<std::size_t> outputShape(const std::vector<std::size_t>& inputShape) const override {
		if (inputShape.size() != 3 || inputShape[0] != inChannels) {
			throw InvalidSizeError(describe() + " needs a [" + std::to_string(inChannels) + ", H, W] input!");
		}
		const std::size_t outHeight = slidingWindowOutputSize(inputShape[1], kernelSize, stride, padding);
		const std::size_t outWidth = slidingWindowOutputSize(inputShape[2], kernelSize, stride, padding);
		if (outHeight == 0 || outWidth == 0) {
			throw InvalidSizeError(describe() + ": the input is smaller than the kernel!");
		}
		return { outChannels, outHeight, outWidth };
	}

	Tensor<T> forward(const Tensor<T>& input, std::size_t threads) override {
		if (input.rank() != 4 || input.shape[0] == 0) {
			throw InvalidSizeError(describe() + " needs a non-empty [N, C, H, W] input!");
		}
		const std::vector<std::size_t> sampleOut = outputShape({ input.shape[1], input.shape[2], input.shape[3] });

		cachedInput = input;
		const std::size_t batch = input.shape[0];
		const std::size_t height = input.shape[2];
		const std::size_t width = input.shape[3];
		const std::size_t outHeight = sampleOut[1];
		const std::size_t outWidth = sampleOut[2];

		Tensor<T> output({ batch, outChannels, outHeight, outWidth });

		// One item = one output feature map (n, oc): it writes only its own plane.
		parallelFor(batch * outChannels, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t plane = begin; plane < end; plane++) {
				const std::size_t n = plane / outChannels;
				const std::size_t oc = plane % outChannels;
				const T* filter = &weights.value[oc * inChannels * kernelSize * kernelSize];
				T* out = &output[plane * outHeight * outWidth];

				for (std::size_t oh = 0; oh < outHeight; oh++) {
					for (std::size_t ow = 0; ow < outWidth; ow++) {
						T sum = bias.value[oc];
						for (std::size_t ic = 0; ic < inChannels; ic++) {
							const T* image = &input[(n * inChannels + ic) * height * width];
							const T* kernel = filter + ic * kernelSize * kernelSize;
							for (std::size_t kh = 0; kh < kernelSize; kh++) {
								const std::ptrdiff_t ih = inputIndex(oh, kh);
								if (ih < 0 || ih >= static_cast<std::ptrdiff_t>(height)) {
									continue; // padding row: contributes 0
								}
								for (std::size_t kw = 0; kw < kernelSize; kw++) {
									const std::ptrdiff_t iw = inputIndex(ow, kw);
									if (iw < 0 || iw >= static_cast<std::ptrdiff_t>(width)) {
										continue;
									}
									sum += kernel[kh * kernelSize + kw] * image[ih * width + iw];
								}
							}
						}
						out[oh * outWidth + ow] = sum;
					}
				}
			}
		});

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

		const Tensor<T>& input = cachedInput;
		const std::size_t batch = input.shape[0];
		const std::size_t height = input.shape[2];
		const std::size_t width = input.shape[3];
		const std::size_t outHeight = gradOutput.shape[2];
		const std::size_t outWidth = gradOutput.shape[3];
		const std::size_t filterSize = inChannels * kernelSize * kernelSize;

		// Weight and bias gradients. One item = one filter oc: it owns
		// weights.grad[oc] and bias.grad[oc]. Each weight sees the gradient of
		// every output position it touched, times the input it was multiplied by:
		//   dW[oc, ic, kh, kw] = sum over n, oh, ow of dOut[n, oc, oh, ow] * in[n, ic, ih, iw]
		//   dB[oc]            = sum over n, oh, ow of dOut[n, oc, oh, ow]
		parallelFor(outChannels, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t oc = begin; oc < end; oc++) {
				T* filterGrad = &weights.grad[oc * filterSize];
				T biasSum = T(0);
				for (std::size_t n = 0; n < batch; n++) {
					const T* dOut = &gradOutput[(n * outChannels + oc) * outHeight * outWidth];
					for (std::size_t oh = 0; oh < outHeight; oh++) {
						for (std::size_t ow = 0; ow < outWidth; ow++) {
							const T g = dOut[oh * outWidth + ow];
							biasSum += g;
							for (std::size_t ic = 0; ic < inChannels; ic++) {
								const T* image = &input[(n * inChannels + ic) * height * width];
								T* kernelGrad = filterGrad + ic * kernelSize * kernelSize;
								for (std::size_t kh = 0; kh < kernelSize; kh++) {
									const std::ptrdiff_t ih = inputIndex(oh, kh);
									if (ih < 0 || ih >= static_cast<std::ptrdiff_t>(height)) {
										continue;
									}
									for (std::size_t kw = 0; kw < kernelSize; kw++) {
										const std::ptrdiff_t iw = inputIndex(ow, kw);
										if (iw < 0 || iw >= static_cast<std::ptrdiff_t>(width)) {
											continue;
										}
										kernelGrad[kh * kernelSize + kw] += g * image[ih * width + iw];
									}
								}
							}
						}
					}
				}
				bias.grad[oc] += biasSum;
			}
		});

		// Input gradient. One item = one sample n: it owns gradInput[n]. Every
		// output position sends its gradient back through the filter weights to
		// the input pixels of its window (a "full" convolution with the filter
		// flipped, written here as a scatter):
		//   dIn[n, ic, ih, iw] += dOut[n, oc, oh, ow] * W[oc, ic, kh, kw]
		Tensor<T> gradInput(input.shape, T(0));
		parallelFor(batch, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t n = begin; n < end; n++) {
				for (std::size_t oc = 0; oc < outChannels; oc++) {
					const T* filter = &weights.value[oc * filterSize];
					const T* dOut = &gradOutput[(n * outChannels + oc) * outHeight * outWidth];
					for (std::size_t oh = 0; oh < outHeight; oh++) {
						for (std::size_t ow = 0; ow < outWidth; ow++) {
							const T g = dOut[oh * outWidth + ow];
							for (std::size_t ic = 0; ic < inChannels; ic++) {
								T* dImage = &gradInput[(n * inChannels + ic) * height * width];
								const T* kernel = filter + ic * kernelSize * kernelSize;
								for (std::size_t kh = 0; kh < kernelSize; kh++) {
									const std::ptrdiff_t ih = inputIndex(oh, kh);
									if (ih < 0 || ih >= static_cast<std::ptrdiff_t>(height)) {
										continue;
									}
									for (std::size_t kw = 0; kw < kernelSize; kw++) {
										const std::ptrdiff_t iw = inputIndex(ow, kw);
										if (iw < 0 || iw >= static_cast<std::ptrdiff_t>(width)) {
											continue;
										}
										dImage[ih * width + iw] += g * kernel[kh * kernelSize + kw];
									}
								}
							}
						}
					}
				}
			}
		});

		return gradInput;
	}

	std::vector<Parameter<T>*> parameters() override {
		return { &weights, &bias };
	}

private:
	std::size_t inChannels;
	std::size_t outChannels;
	std::size_t kernelSize;
	std::size_t stride;
	std::size_t padding;

	Tensor<T> cachedInput;
	std::vector<std::size_t> cachedOutputShape;

	/**
	 * @brief Input row (or column) under kernel offset k at output position o;
	 * negative or past the edge means a padding position.
	 */
	std::ptrdiff_t inputIndex(std::size_t o, std::size_t k) const {
		return static_cast<std::ptrdiff_t>(o * stride + k) - static_cast<std::ptrdiff_t>(padding);
	}
};
