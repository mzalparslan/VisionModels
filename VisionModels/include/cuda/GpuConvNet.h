#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "Exceptions.h"
#include "LayerSpec.h"
#include "Optimizer.h"

namespace cuda {

	/**
	 * @brief A ConvNet that lives on the GPU: the same layers as the CPU
	 * ConvNet (Conv2D, ReLU, MaxPool2D, Flatten, Dense, softmax cross-entropy),
	 * with every forward, backward and optimizer step done by CUDA kernels.
	 *
	 * Build it from the CPU model's ConvNet::specs(), copy the weights up with
	 * uploadParameters(), train, and copy them back down with
	 * downloadParameters(). While training runs the weights, gradients,
	 * optimizer state and every activation stay in GPU memory. With
	 * uploadTrainingSet() the images stay there too, so a training step sends
	 * only the batch's image indices up and one loss per image back.
	 *
	 * Kernels use one GPU thread per output number (per output pixel, per
	 * weight gradient, ...), mirroring the CPU loops so the two are easy to
	 * compare. They are written to be read, not to compete with cuDNN.
	 *
	 * float only: consumer GPUs run double at a small fraction of the speed.
	 * Results match the CPU in float to within rounding.
	 *
	 * Plain C++ header; the implementation is in the VisionModels.Cuda project.
	 * Without the CUDA Toolkit the constructor throws CudaError.
	 */
	class GpuConvNet {
	public:
		/**
		 * @param layers ConvNet::specs() of the model.
		 * @param inputShape One sample's shape, [channels, height, width].
		 * @param maxBatch Largest batch a training step will use (memory for
		 * activations is sized for it).
		 * @throws InvalidSizeError If the layers do not fit together or maxBatch is 0.
		 * @throws CudaError If CUDA is not available or the GPU runs out of memory.
		 */
		GpuConvNet(const std::vector<LayerSpec>& layers, const std::vector<std::size_t>& inputShape, std::size_t maxBatch);
		~GpuConvNet();

		GpuConvNet(GpuConvNet&&) noexcept;
		GpuConvNet& operator=(GpuConvNet&&) noexcept;
		GpuConvNet(const GpuConvNet&) = delete;
		GpuConvNet& operator=(const GpuConvNet&) = delete;

		std::size_t maxBatch() const;
		std::size_t classCount() const;

		/**
		 * @brief Copies every parameter to the GPU (in ConvNet::parameters()
		 * order: each layer's weights, then its bias), clears optimizer state
		 * and sets the step count.
		 *
		 * @throws InvalidSizeError If the count or a size differs from the layers.
		 */
		void uploadParameters(const std::vector<std::vector<float>>& values, std::size_t stepsTaken = 0);

		/**
		 * @brief Copies every parameter back to the host, in the same order.
		 */
		void downloadParameters(std::vector<std::vector<float>>& values) const;

		/**
		 * @brief Copies the gradients of the last computeGradients...() call
		 * to the host, in the same order as the parameters.
		 */
		void downloadGradients(std::vector<std::vector<float>>& gradients) const;

		/**
		 * @brief Keeps a whole dataset in GPU memory for
		 * computeGradientsResident() (MNIST's 60,000 images take 188 MB).
		 *
		 * @param pixels [count, C, H, W] images.
		 * @param labels count class indices.
		 * @throws InvalidSizeError If the sizes disagree with each other or the input shape.
		 */
		void uploadTrainingSet(const std::vector<float>& pixels, const std::vector<std::uint8_t>& labels);

		std::size_t trainingSetSize() const;

		/**
		 * @brief Forward, loss and backward for a batch sent from the host.
		 * Afterwards the GPU holds dLoss/dWeights (see downloadGradients()).
		 *
		 * @param images [N, C, H, W] with N = labels.size().
		 * @return Mean cross-entropy loss of the batch.
		 * @throws InvalidSizeError If N is 0, above maxBatch(), or images has the wrong size.
		 * @throws InvalidParameterError If a label is not below classCount().
		 * @throws NaNError, NonFiniteError If the loss is not finite.
		 */
		float computeGradients(const std::vector<float>& images, const std::vector<std::size_t>& labels);

		/**
		 * @brief computeGradients() for the images at `indices` of the uploaded
		 * training set: only the indices cross to the GPU.
		 *
		 * @throws InvalidSizeError If no training set was uploaded, or indices is
		 * empty or longer than maxBatch().
		 * @throws InvalidParameterError If an index is out of range.
		 */
		float computeGradientsResident(const std::vector<std::uint32_t>& indices);

		/**
		 * @brief One optimizer step on every parameter, the same formulas as
		 * Parameter::update().
		 *
		 * @throws InvalidParameterError If the settings are invalid.
		 */
		void update(const OptimizerSettings& optimizer);

		/**
		 * @brief Optimizer steps taken so far.
		 */
		std::size_t steps() const;

		/**
		 * @brief Class scores for any number of images (sent in batches of maxBatch()).
		 *
		 * @param images [count, C, H, W].
		 * @param result [count, classes] logits.
		 * @throws InvalidSizeError If images has the wrong size.
		 */
		void logits(const std::vector<float>& images, std::size_t count, std::vector<float>& result);

	private:
		class Impl;
		std::unique_ptr<Impl> impl;
	};

} // namespace cuda
