#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "LayerSpec.h"
#include "Parameter.h"
#include "Tensor.h"

/**
 * @brief One layer of a ConvNet: a forward pass and its backward pass.
 *
 * Tensors carry the batch as their first dimension: images are
 * [N, C, H, W] (NCHW), feature vectors are [N, F].
 *
 * forward() keeps what backward() will need (its input, or which input won a
 * max pool window), so the two must be called in pairs: forward on a batch,
 * then backward with the gradient of the loss with respect to that forward's
 * output. backward() adds (+=) the weight gradients into parameters()[i].grad
 * and returns the gradient with respect to the input, for the layer before.
 *
 * Both take `threads`: 1 runs on the calling thread (Sequential), more splits
 * the work across ThreadPool::shared() (Parallel), with identical results.
 */
template <typename T>
class Layer {
public:
	virtual ~Layer() = default;

	/**
	 * @brief Kind and sizes of this layer, e.g. to rebuild it on the GPU.
	 */
	virtual LayerSpec spec() const = 0;

	/**
	 * @brief One line for a model summary, e.g. "Conv2D(1 -> 8, 3x3, stride 1, padding 1)".
	 */
	virtual std::string describe() const = 0;

	/**
	 * @brief Shape of one sample's output for one sample's input shape
	 * ([C, H, W] or [F], without the batch dimension).
	 *
	 * @throws InvalidSizeError If this layer cannot take that input.
	 */
	virtual std::vector<std::size_t> outputShape(const std::vector<std::size_t>& inputShape) const = 0;

	/**
	 * @throws InvalidSizeError If input has the wrong shape.
	 */
	virtual Tensor<T> forward(const Tensor<T>& input, std::size_t threads) = 0;

	/**
	 * @throws InvalidSizeError If gradOutput's shape differs from the last forward output.
	 * @throws PipelineStateError If forward() has not been called.
	 */
	virtual Tensor<T> backward(const Tensor<T>& gradOutput, std::size_t threads) = 0;

	/**
	 * @brief Learnable parameters (weights then bias), empty for layers without any.
	 */
	virtual std::vector<Parameter<T>*> parameters() { return {}; }
};
