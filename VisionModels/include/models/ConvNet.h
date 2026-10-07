#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Conv2D.h"
#include "Dense.h"
#include "Flatten.h"
#include "Layer.h"
#include "MaxPool2D.h"
#include "Optimizer.h"
#include "ReLU.h"
#include "SoftmaxCrossEntropy.h"
#include "Validation.h"

/**
 * @brief A convolutional network: a stack of layers run in order, ending in
 * one score (logit) per class, trained with softmax cross-entropy.
 *
 * @code
 * RandomEngine rng(42);
 * ConvNet<float> net({ 1, 28, 28 });           // one channel, 28x28 pixels
 * net.addConv2D(8, 3, 1, 1, rng);              // -> [8, 28, 28]
 * net.addReLU();
 * net.addMaxPool2D(2, 2);                      // -> [8, 14, 14]
 * net.addFlatten();                            // -> [1568]
 * net.addDense(10, rng);                       // -> 10 class scores
 * float loss = net.trainBatch(images, labels, OptimizerSettings::adam(1e-3));
 * @endcode
 *
 * Each add...() method works out the new layer's input size from the layer
 * before, and checks that it fits, so a mistake shows up when the network is
 * built rather than in the middle of training.
 *
 * `threads` (default 1) is how the CPU strategies are chosen: 1 is
 * Sequential, more is Parallel with identical results. The GPU strategy uses
 * cuda::GpuConvNet, built from specs() and parameterValues().
 */
template <typename T>
class ConvNet {
public:
	/**
	 * @param inputShape One sample's shape, [channels, height, width].
	 * @throws InvalidSizeError If inputShape is not 3 non-zero sizes.
	 */
	explicit ConvNet(const std::vector<std::size_t>& inputShape)
		: sampleInputShape(inputShape), currentShape(inputShape)
	{
		if (inputShape.size() != 3) {
			throw InvalidSizeError("ConvNet input shape must be [channels, height, width]!");
		}
		validation::requireValidShape(inputShape, "ConvNet input shape");
	}

	ConvNet(ConvNet&&) noexcept = default;
	ConvNet& operator=(ConvNet&&) noexcept = default;
	ConvNet(const ConvNet&) = delete;
	ConvNet& operator=(const ConvNet&) = delete;

	/**
	 * @brief Adds a Conv2D whose input channels are the current channel count.
	 * @throws InvalidSizeError If the current output is not an image or is smaller than the kernel.
	 */
	ConvNet& addConv2D(std::size_t outChannels, std::size_t kernelSize, std::size_t stride,
		std::size_t padding, RandomEngine& rng) {
		if (currentShape.size() != 3) {
			throw InvalidSizeError("Conv2D must come before Flatten!");
		}
		return add(std::make_unique<Conv2D<T>>(currentShape[0], outChannels, kernelSize, stride, padding, rng));
	}

	ConvNet& addReLU() {
		return add(std::make_unique<ReLU<T>>());
	}

	ConvNet& addMaxPool2D(std::size_t kernelSize, std::size_t stride) {
		return add(std::make_unique<MaxPool2D<T>>(kernelSize, stride));
	}

	ConvNet& addFlatten() {
		return add(std::make_unique<Flatten<T>>());
	}

	/**
	 * @brief Adds a Dense layer whose input size is the current feature count.
	 * @throws InvalidSizeError If the current output is still an image (add Flatten first).
	 */
	ConvNet& addDense(std::size_t outFeatures, RandomEngine& rng) {
		if (currentShape.size() != 1) {
			throw InvalidSizeError("Dense needs a Flatten layer before it!");
		}
		return add(std::make_unique<Dense<T>>(currentShape[0], outFeatures, rng));
	}

	/**
	 * @brief Adds any layer after checking that it accepts the current output.
	 * @throws InvalidSizeError If it does not.
	 */
	ConvNet& add(std::unique_ptr<Layer<T>> layer) {
		std::vector<std::size_t> next = layer->outputShape(currentShape);
		layers.push_back(std::move(layer));
		currentShape = std::move(next);
		return *this;
	}

	const std::vector<std::size_t>& inputShape() const { return sampleInputShape; }

	/**
	 * @brief One sample's output shape so far ([classes] once the net is complete).
	 */
	const std::vector<std::size_t>& outputShape() const { return currentShape; }

	/**
	 * @brief Number of values the network outputs per sample.
	 * @throws InvalidSizeError If it does not end in a feature vector yet.
	 */
	std::size_t classCount() const {
		if (currentShape.size() != 1) {
			throw InvalidSizeError("ConvNet must end in a Dense layer to classify!");
		}
		return currentShape[0];
	}

	std::size_t layerCount() const { return layers.size(); }
	Layer<T>& layer(std::size_t index) { return *layers.at(index); }

	/**
	 * @brief Kind and sizes of every layer, in order.
	 */
	std::vector<LayerSpec> specs() const {
		std::vector<LayerSpec> result;
		for (const auto& layerPointer : layers) {
			result.push_back(layerPointer->spec());
		}
		return result;
	}

	/**
	 * @brief Every learnable parameter in order: each Conv2D and Dense layer's
	 * weights, then its bias.
	 */
	std::vector<Parameter<T>*> parameters() {
		std::vector<Parameter<T>*> result;
		for (auto& layerPointer : layers) {
			for (Parameter<T>* parameter : layerPointer->parameters()) {
				result.push_back(parameter);
			}
		}
		return result;
	}

	std::size_t parameterCount() {
		std::size_t count = 0;
		for (Parameter<T>* parameter : parameters()) {
			count += parameter->value.size();
		}
		return count;
	}

	/**
	 * @brief Logits [N, classes] for a batch of images [N, C, H, W].
	 * @throws InvalidSizeError If the images do not match inputShape().
	 */
	Tensor<T> forward(const Tensor<T>& images, std::size_t threads = 1) {
		classCount();
		if (images.rank() != 4 || images.shape[0] == 0
			|| std::vector<std::size_t>(images.shape.begin() + 1, images.shape.end()) != sampleInputShape) {
			throw InvalidSizeError("ConvNet input must be a non-empty [N, " + shapeText(sampleInputShape) + "] batch!");
		}
		Tensor<T> activation = images;
		for (auto& layerPointer : layers) {
			activation = layerPointer->forward(activation, threads);
		}
		return activation;
	}

	/**
	 * @brief Runs every layer's backward pass, last to first, adding the
	 * weight gradients into each parameter's grad.
	 */
	void backward(const Tensor<T>& gradLogits, std::size_t threads = 1) {
		Tensor<T> gradient = gradLogits;
		for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
			gradient = (*it)->backward(gradient, threads);
		}
	}

	void zeroGrad() {
		for (Parameter<T>* parameter : parameters()) {
			parameter->zeroGrad();
		}
	}

	/**
	 * @brief Forward, loss and backward on a batch: afterwards every
	 * parameter's grad holds dLoss/dWeights. Does not change the weights.
	 *
	 * @return Mean cross-entropy loss of the batch.
	 */
	T computeGradients(const Tensor<T>& images, const std::vector<std::size_t>& labels, std::size_t threads = 1) {
		zeroGrad();
		Tensor<T> logits = forward(images, threads);
		Tensor<T> gradLogits;
		const T loss = SoftmaxCrossEntropy<T>::lossAndGradient(logits, labels, gradLogits);
		backward(gradLogits, threads);
		return loss;
	}

	/**
	 * @brief One optimizer step on every parameter using the current gradients.
	 */
	void applyUpdate(const OptimizerSettings& optimizer) {
		optimizer.validate();
		stepCount++;
		for (Parameter<T>* parameter : parameters()) {
			parameter->update(optimizer, stepCount);
		}
	}

	/**
	 * @brief One training step: computeGradients() then applyUpdate().
	 * @return Mean loss of the batch before the update.
	 */
	T trainBatch(const Tensor<T>& images, const std::vector<std::size_t>& labels,
		const OptimizerSettings& optimizer, std::size_t threads = 1) {
		const T loss = computeGradients(images, labels, threads);
		applyUpdate(optimizer);
		return loss;
	}

	/**
	 * @brief Most likely class of every image.
	 */
	std::vector<std::size_t> predict(const Tensor<T>& images, std::size_t threads = 1) {
		return argmaxRows(forward(images, threads));
	}

	/**
	 * @brief Index of the largest value in each row of [N, classes] scores.
	 */
	static std::vector<std::size_t> argmaxRows(const Tensor<T>& scores) {
		validation::requireMatrix(scores, "Scores");
		const std::size_t classes = scores.shape[1];
		std::vector<std::size_t> result(scores.shape[0]);
		for (std::size_t n = 0; n < result.size(); n++) {
			const T* row = &scores[n * classes];
			result[n] = static_cast<std::size_t>(std::max_element(row, row + classes) - row);
		}
		return result;
	}

	/**
	 * @brief Optimizer steps taken so far (Adam's timestep).
	 */
	std::size_t steps() const { return stepCount; }

	/**
	 * @brief Copies every parameter out, converted to U, in parameters() order.
	 */
	template <typename U>
	std::vector<std::vector<U>> parameterValues() {
		std::vector<std::vector<U>> result;
		for (Parameter<T>* parameter : parameters()) {
			result.emplace_back(parameter->value.data.begin(), parameter->value.data.end());
		}
		return result;
	}

	/**
	 * @brief Replaces every parameter, in parameters() order, and clears
	 * optimizer state (it belonged to the old weights).
	 *
	 * @param stepsTaken What steps() should report afterwards.
	 * @throws InvalidSizeError If the count or a size differs.
	 */
	template <typename U>
	void setParameterValues(const std::vector<std::vector<U>>& values, std::size_t stepsTaken) {
		std::vector<Parameter<T>*> targets = parameters();
		validation::requireSameSize(values.size(), targets.size(), "Parameter list");
		for (std::size_t p = 0; p < targets.size(); p++) {
			validation::requireSameSize(values[p].size(), targets[p]->value.size(), "Parameter");
		}
		for (std::size_t p = 0; p < targets.size(); p++) {
			std::transform(values[p].begin(), values[p].end(), targets[p]->value.data.begin(),
				[](U x) { return static_cast<T>(x); });
			targets[p]->resetOptimizerState();
		}
		stepCount = stepsTaken;
	}

	/**
	 * @brief Layer-by-layer description with output shapes and parameter counts.
	 */
	std::string summary() {
		std::ostringstream text;
		std::vector<std::size_t> shape = sampleInputShape;
		text << "Input                                     [" << shapeText(shape) << "]\n";
		for (auto& layerPointer : layers) {
			shape = layerPointer->outputShape(shape);
			std::size_t count = 0;
			for (Parameter<T>* parameter : layerPointer->parameters()) {
				count += parameter->value.size();
			}
			std::string line = layerPointer->describe();
			line.resize(std::max<std::size_t>(line.size(), 42), ' ');
			text << line << "[" << shapeText(shape) << "]";
			if (count > 0) {
				text << "  " << count << " parameters";
			}
			text << "\n";
		}
		text << "Total parameters: " << parameterCount() << "\n";
		return text.str();
	}

private:
	std::vector<std::unique_ptr<Layer<T>>> layers;
	std::vector<std::size_t> sampleInputShape;
	std::vector<std::size_t> currentShape;
	std::size_t stepCount = 0;

	static std::string shapeText(const std::vector<std::size_t>& shape) {
		std::string text;
		for (std::size_t i = 0; i < shape.size(); i++) {
			text += (i ? ", " : "") + std::to_string(shape[i]);
		}
		return text;
	}
};
