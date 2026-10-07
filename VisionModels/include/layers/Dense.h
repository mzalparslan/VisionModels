#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Layer.h"
#include "ParallelFor.h"
#include "Validation.h"

/**
 * @brief Fully connected layer: out = in * W^T + b.
 *
 * Input [N, inFeatures], output [N, outFeatures]. weights is
 * [outFeatures, inFeatures] (one row per output, the PyTorch layout), so
 * each output is a dot product with one contiguous row:
 *
 *   out[n, o] = bias[o] + sum over i of in[n, i] * weights[o, i]
 *
 * In a CNN these come last: they combine the features the convolutions found
 * anywhere in the image into one score per class.
 */
template <typename T>
class Dense : public Layer<T> {
public:
	// [outFeatures, inFeatures]
	Parameter<T> weights;
	// [outFeatures]
	Parameter<T> bias;

	/**
	 * Weights start from He initialization, N(0, 2 / inFeatures); biases at 0.
	 *
	 * @throws InvalidParameterSizeError If a size is 0.
	 */
	Dense(std::size_t inFeatures, std::size_t outFeatures, RandomEngine& rng)
		: inFeatures(inFeatures), outFeatures(outFeatures)
	{
		validation::requirePositiveSize(inFeatures, "Dense input features");
		validation::requirePositiveSize(outFeatures, "Dense output features");

		weights.initNormal({ outFeatures, inFeatures }, std::sqrt(T(2) / static_cast<T>(inFeatures)), rng);
		bias.initConstant({ outFeatures }, T(0));
	}

	LayerSpec spec() const override {
		return LayerSpec::dense(inFeatures, outFeatures);
	}

	std::string describe() const override {
		return "Dense(" + std::to_string(inFeatures) + " -> " + std::to_string(outFeatures) + ")";
	}

	std::vector<std::size_t> outputShape(const std::vector<std::size_t>& inputShape) const override {
		if (inputShape.size() != 1 || inputShape[0] != inFeatures) {
			throw InvalidSizeError(describe() + " needs a [" + std::to_string(inFeatures) + "] input!");
		}
		return { outFeatures };
	}

	Tensor<T> forward(const Tensor<T>& input, std::size_t threads) override {
		if (input.rank() != 2 || input.shape[0] == 0) {
			throw InvalidSizeError(describe() + " needs a non-empty [N, F] input!");
		}
		outputShape({ input.shape[1] });

		cachedInput = input;
		const std::size_t batch = input.shape[0];
		Tensor<T> output({ batch, outFeatures });

		// One item = one sample.
		parallelFor(batch, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t n = begin; n < end; n++) {
				const T* x = &input[n * inFeatures];
				for (std::size_t o = 0; o < outFeatures; o++) {
					const T* row = &weights.value[o * inFeatures];
					T sum = bias.value[o];
					for (std::size_t i = 0; i < inFeatures; i++) {
						sum += x[i] * row[i];
					}
					output[n * outFeatures + o] = sum;
				}
			}
		});
		return output;
	}

	Tensor<T> backward(const Tensor<T>& gradOutput, std::size_t threads) override {
		if (cachedInput.shape.empty()) {
			throw PipelineStateError(describe() + ": backward() called before forward()!");
		}
		const std::size_t batch = cachedInput.shape[0];
		if (gradOutput.shape != std::vector<std::size_t>{ batch, outFeatures }) {
			throw InvalidSizeError(describe() + ": gradient shape differs from the forward output!");
		}

		// dW[o, i] = sum over n of dOut[n, o] * in[n, i];  dB[o] = sum over n of dOut[n, o].
		// One item = one output o: it owns row o of the gradients.
		parallelFor(outFeatures, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t o = begin; o < end; o++) {
				T* rowGrad = &weights.grad[o * inFeatures];
				T biasSum = T(0);
				for (std::size_t n = 0; n < batch; n++) {
					const T g = gradOutput[n * outFeatures + o];
					const T* x = &cachedInput[n * inFeatures];
					biasSum += g;
					for (std::size_t i = 0; i < inFeatures; i++) {
						rowGrad[i] += g * x[i];
					}
				}
				bias.grad[o] += biasSum;
			}
		});

		// dIn[n, i] = sum over o of dOut[n, o] * W[o, i]. One item = one sample.
		Tensor<T> gradInput({ batch, inFeatures }, T(0));
		parallelFor(batch, 1, threads, [&](std::size_t begin, std::size_t end) {
			for (std::size_t n = begin; n < end; n++) {
				T* dx = &gradInput[n * inFeatures];
				for (std::size_t o = 0; o < outFeatures; o++) {
					const T g = gradOutput[n * outFeatures + o];
					const T* row = &weights.value[o * inFeatures];
					for (std::size_t i = 0; i < inFeatures; i++) {
						dx[i] += g * row[i];
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
	std::size_t inFeatures;
	std::size_t outFeatures;

	Tensor<T> cachedInput;
};
