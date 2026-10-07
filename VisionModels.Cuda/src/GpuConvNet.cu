// The GPU implementation of GpuConvNet.h. Compiled by nvcc (see nvcc-build.cmd);
// only built when the CUDA Toolkit is installed.

#include "GpuConvNet.h"
#include "CudaRuntime.h"
#include "detail/CudaHost.cuh"
#include "detail/DeviceArray.cuh"
#include "detail/ConvNetKernels.cuh"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace cuda {

	using detail::blocksFor;
	using detail::blockSize;
	using detail::check;
	using detail::DeviceArray;

	namespace {

		/**
		 * @brief A weight tensor or bias on the GPU, with its gradient and optimizer state.
		 */
		class DeviceParameter {
		public:
			std::size_t count = 0;
			DeviceArray<float> value;
			DeviceArray<float> grad;
			// Momentum's velocity or Adam's m, and Adam's v; allocated on first use.
			DeviceArray<float> firstMoment;
			DeviceArray<float> secondMoment;

			explicit DeviceParameter(std::size_t count)
				: count(count), value(count), grad(count) {
				grad.zero();
			}

			void resetOptimizerState() {
				firstMoment = DeviceArray<float>();
				secondMoment = DeviceArray<float>();
			}

			void ensureMoments(bool needSecond) {
				if (firstMoment.size() != count) {
					firstMoment.allocate(count);
					firstMoment.zero();
				}
				if (needSecond && secondMoment.size() != count) {
					secondMoment.allocate(count);
					secondMoment.zero();
				}
			}
		};

		/**
		 * @brief One layer on the GPU: its sizes, which activation buffers it
		 * reads and writes, and its parameters (weights then bias).
		 */
		class DeviceLayer {
		public:
			LayerSpec spec;
			// One sample's input and output shape: [C, H, W], or [F] with C = F, H = W = 1.
			int inChannels = 0, inHeight = 1, inWidth = 1;
			int outChannels = 0, outHeight = 1, outWidth = 1;
			std::size_t inputBuffer = 0;
			std::size_t outputBuffer = 0;
			std::vector<DeviceParameter> parameters;
			// MaxPool2D: flat input index of each output's winner.
			DeviceArray<int> winners;

			std::size_t inputSize() const { return static_cast<std::size_t>(inChannels) * inHeight * inWidth; }
			std::size_t outputSize() const { return static_cast<std::size_t>(outChannels) * outHeight * outWidth; }
		};

		void requireFiniteLoss(float loss) {
			if (std::isnan(loss)) {
				throw NaNError("GpuConvNet: cross-entropy loss is NaN!");
			}
			if (!std::isfinite(loss)) {
				throw NonFiniteError("GpuConvNet: cross-entropy loss is infinite!");
			}
		}

	} // namespace

	class GpuConvNet::Impl {
	public:
		std::vector<DeviceLayer> layers;
		std::vector<std::size_t> inputShape;
		std::size_t batchCapacity = 0;
		std::size_t classes = 0;
		std::size_t stepCount = 0;

		// activations[b] holds buffer b for a whole batch; buffer 0 is the input
		// images. gradients[b] is dLoss/d(activations[b]).
		std::vector<DeviceArray<float>> activations;
		std::vector<DeviceArray<float>> gradients;
		std::vector<std::size_t> bufferSampleSizes;
		DeviceArray<int> batchLabels;
		DeviceArray<float> sampleLosses;
		// First stage of the convolution parameter gradients, [batch, parameters].
		DeviceArray<float> partialSums;

		// The resident training set.
		DeviceArray<float> setPixels;
		DeviceArray<int> setLabels;
		std::size_t setSize = 0;
		DeviceArray<unsigned> batchIndices;

		Impl(const std::vector<LayerSpec>& specs, const std::vector<std::size_t>& shape, std::size_t maxBatch)
			: inputShape(shape), batchCapacity(maxBatch) {
			if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
				throw InvalidSizeError("GpuConvNet: input shape must be [channels, height, width]!");
			}
			if (maxBatch == 0) {
				throw InvalidSizeError("GpuConvNet: maxBatch must be greater than zero!");
			}
			if (specs.empty()) {
				throw InvalidSizeError("GpuConvNet: the network has no layers!");
			}
			if (!isAvailable()) {
				throw CudaError("GpuConvNet: No CUDA device is available.");
			}

			// Walk the layers, working out shapes the same way the CPU layers do.
			int channels = static_cast<int>(shape[0]);
			int height = static_cast<int>(shape[1]);
			int width = static_cast<int>(shape[2]);
			bool isImage = true;
			bufferSampleSizes.push_back(shape[0] * shape[1] * shape[2]);
			std::size_t maxPartial = 0;

			for (const LayerSpec& spec : specs) {
				DeviceLayer layer;
				layer.spec = spec;
				layer.inChannels = channels;
				layer.inHeight = height;
				layer.inWidth = width;
				layer.inputBuffer = bufferSampleSizes.size() - 1;

				switch (spec.kind) {
				case LayerKind::Conv2D: {
					if (!isImage || spec.inChannels != static_cast<std::size_t>(channels)) {
						throw InvalidSizeError("GpuConvNet: a Conv2D layer does not match its input!");
					}
					const std::size_t outH = slidingWindowOutputSize(height, spec.kernelSize, spec.stride, spec.padding);
					const std::size_t outW = slidingWindowOutputSize(width, spec.kernelSize, spec.stride, spec.padding);
					if (outH == 0 || outW == 0) {
						throw InvalidSizeError("GpuConvNet: a Conv2D input is smaller than its kernel!");
					}
					channels = static_cast<int>(spec.outChannels);
					height = static_cast<int>(outH);
					width = static_cast<int>(outW);
					const std::size_t weightCount = spec.outChannels * spec.inChannels * spec.kernelSize * spec.kernelSize;
					layer.parameters.emplace_back(weightCount);
					layer.parameters.emplace_back(spec.outChannels);
					maxPartial = std::max(maxPartial, weightCount + spec.outChannels);
					break;
				}
				case LayerKind::ReLU:
					break;
				case LayerKind::MaxPool2D: {
					if (!isImage) {
						throw InvalidSizeError("GpuConvNet: MaxPool2D needs an image input!");
					}
					const std::size_t outH = slidingWindowOutputSize(height, spec.kernelSize, spec.stride, 0);
					const std::size_t outW = slidingWindowOutputSize(width, spec.kernelSize, spec.stride, 0);
					if (outH == 0 || outW == 0) {
						throw InvalidSizeError("GpuConvNet: a MaxPool2D input is smaller than its window!");
					}
					height = static_cast<int>(outH);
					width = static_cast<int>(outW);
					break;
				}
				case LayerKind::Flatten:
					channels = channels * height * width;
					height = 1;
					width = 1;
					isImage = false;
					break;
				case LayerKind::Dense:
					if (isImage || spec.inFeatures != static_cast<std::size_t>(channels)) {
						throw InvalidSizeError("GpuConvNet: a Dense layer does not match its input!");
					}
					channels = static_cast<int>(spec.outFeatures);
					layer.parameters.emplace_back(spec.outFeatures * spec.inFeatures);
					layer.parameters.emplace_back(spec.outFeatures);
					break;
				}

				layer.outChannels = channels;
				layer.outHeight = height;
				layer.outWidth = width;
				if (spec.kind == LayerKind::Flatten) {
					// Same numbers, new shape: share the buffer.
					layer.outputBuffer = layer.inputBuffer;
				}
				else {
					layer.outputBuffer = bufferSampleSizes.size();
					bufferSampleSizes.push_back(layer.outputSize());
				}
				if (spec.kind == LayerKind::MaxPool2D) {
					layer.winners.allocate(maxBatch * layer.outputSize());
				}
				layers.push_back(std::move(layer));
			}

			if (isImage || height != 1 || width != 1) {
				throw InvalidSizeError("GpuConvNet: the network must end in a feature vector (Flatten, then Dense)!");
			}
			classes = static_cast<std::size_t>(channels);

			for (std::size_t sampleSize : bufferSampleSizes) {
				activations.emplace_back(maxBatch * sampleSize);
				gradients.emplace_back(maxBatch * sampleSize);
			}
			batchLabels.allocate(maxBatch);
			sampleLosses.allocate(maxBatch);
			batchIndices.allocate(maxBatch);
			partialSums.allocate(maxBatch * maxPartial);
		}

		std::vector<DeviceParameter*> allParameters() {
			std::vector<DeviceParameter*> result;
			for (DeviceLayer& layer : layers) {
				for (DeviceParameter& parameter : layer.parameters) {
					result.push_back(&parameter);
				}
			}
			return result;
		}

		/**
		 * @brief Runs every layer on the batch already in activations[0].
		 */
		void forward(std::size_t batch) {
			for (DeviceLayer& layer : layers) {
				const float* input = activations[layer.inputBuffer].get();
				float* output = activations[layer.outputBuffer].get();
				const std::size_t total = batch * layer.outputSize();

				switch (layer.spec.kind) {
				case LayerKind::Conv2D:
					kernels::conv2dForward<<<blocksFor(total), blockSize>>>(total, input,
						layer.parameters[0].value.get(), layer.parameters[1].value.get(), output,
						layer.inChannels, layer.inHeight, layer.inWidth, layer.outChannels, layer.outHeight, layer.outWidth,
						static_cast<int>(layer.spec.kernelSize), static_cast<int>(layer.spec.stride), static_cast<int>(layer.spec.padding));
					break;
				case LayerKind::ReLU:
					kernels::reluForward<<<blocksFor(total), blockSize>>>(total, input, output);
					break;
				case LayerKind::MaxPool2D:
					kernels::maxPoolForward<<<blocksFor(total), blockSize>>>(total, input, output, layer.winners.get(),
						layer.inHeight, layer.inWidth, layer.outHeight, layer.outWidth,
						static_cast<int>(layer.spec.kernelSize), static_cast<int>(layer.spec.stride));
					break;
				case LayerKind::Flatten:
					break;
				case LayerKind::Dense:
					kernels::denseForward<<<blocksFor(total), blockSize>>>(total, input,
						layer.parameters[0].value.get(), layer.parameters[1].value.get(), output,
						layer.inChannels, layer.outChannels);
					break;
				}
				check(cudaGetLastError(), "forward kernel launch");
			}
		}

		/**
		 * @brief Loss and backward pass for the batch whose forward just ran,
		 * with its labels in batchLabels.
		 */
		float lossAndBackward(std::size_t batch) {
			const std::size_t logitsBuffer = layers.back().outputBuffer;
			kernels::softmaxCrossEntropy<<<blocksFor(batch), blockSize>>>(batch, activations[logitsBuffer].get(),
				batchLabels.get(), gradients[logitsBuffer].get(), sampleLosses.get(), static_cast<int>(classes));
			check(cudaGetLastError(), "softmaxCrossEntropy kernel launch");

			for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
				DeviceLayer& layer = *it;
				const float* input = activations[layer.inputBuffer].get();
				const float* gradOutput = gradients[layer.outputBuffer].get();
				float* gradInput = gradients[layer.inputBuffer].get();
				// Nothing needs the gradient of the input images.
				const bool needInputGradient = layer.inputBuffer != 0;
				const std::size_t inputTotal = batch * layer.inputSize();
				const std::size_t outputTotal = batch * layer.outputSize();
				const int kernelSize = static_cast<int>(layer.spec.kernelSize);
				const int stride = static_cast<int>(layer.spec.stride);
				const int padding = static_cast<int>(layer.spec.padding);

				switch (layer.spec.kind) {
				case LayerKind::Conv2D: {
					DeviceParameter& weights = layer.parameters[0];
					DeviceParameter& bias = layer.parameters[1];
					const std::size_t perSample = weights.count + bias.count;
					const std::size_t partialTotal = batch * perSample;
					kernels::conv2dBackwardParametersPartial<<<blocksFor(partialTotal), blockSize>>>(partialTotal, input,
						gradOutput, partialSums.get(), layer.inChannels, layer.inHeight, layer.inWidth,
						layer.outChannels, layer.outHeight, layer.outWidth, kernelSize, stride, padding);
					check(cudaGetLastError(), "conv2dBackwardParametersPartial kernel launch");
					kernels::reduceBatch<<<blocksFor(perSample), blockSize>>>(perSample, batch, partialSums.get(),
						weights.count, weights.grad.get(), bias.grad.get());
					check(cudaGetLastError(), "reduceBatch kernel launch");
					if (needInputGradient) {
						kernels::conv2dBackwardInput<<<blocksFor(inputTotal), blockSize>>>(inputTotal, gradOutput,
							weights.value.get(), gradInput, layer.inChannels, layer.inHeight, layer.inWidth,
							layer.outChannels, layer.outHeight, layer.outWidth, kernelSize, stride, padding);
					}
					break;
				}
				case LayerKind::ReLU:
					if (needInputGradient) {
						kernels::reluBackward<<<blocksFor(inputTotal), blockSize>>>(inputTotal, input, gradOutput, gradInput);
					}
					break;
				case LayerKind::MaxPool2D:
					if (needInputGradient) {
						check(cudaMemset(gradInput, 0, inputTotal * sizeof(float)), "cudaMemset");
						kernels::maxPoolBackward<<<blocksFor(outputTotal), blockSize>>>(outputTotal, gradOutput,
							layer.winners.get(), gradInput);
					}
					break;
				case LayerKind::Flatten:
					// Shares its buffer with the layer before: the gradient is already there.
					break;
				case LayerKind::Dense: {
					DeviceParameter& weights = layer.parameters[0];
					DeviceParameter& bias = layer.parameters[1];
					const std::size_t parameterTotal = weights.count + bias.count;
					kernels::denseBackwardParameters<<<blocksFor(parameterTotal), blockSize>>>(parameterTotal, input,
						gradOutput, weights.grad.get(), bias.grad.get(), batch, layer.inChannels, layer.outChannels);
					check(cudaGetLastError(), "denseBackwardParameters kernel launch");
					if (needInputGradient) {
						kernels::denseBackwardInput<<<blocksFor(inputTotal), blockSize>>>(inputTotal, gradOutput,
							weights.value.get(), gradInput, layer.inChannels, layer.outChannels);
					}
					break;
				}
				}
				check(cudaGetLastError(), "backward kernel launch");
			}

			// The mean loss, summed on the host in a fixed order.
			std::vector<float> losses(batch);
			sampleLosses.download(losses.data(), batch);
			double total = 0.0;
			for (float loss : losses) {
				total += loss;
			}
			const float mean = static_cast<float>(total / static_cast<double>(batch));
			requireFiniteLoss(mean);
			return mean;
		}

		void requireBatch(std::size_t batch) const {
			if (batch == 0 || batch > batchCapacity) {
				throw InvalidSizeError("GpuConvNet: batch size " + std::to_string(batch)
					+ " must be between 1 and maxBatch (" + std::to_string(batchCapacity) + ")!");
			}
		}
	};

	GpuConvNet::GpuConvNet(const std::vector<LayerSpec>& layers, const std::vector<std::size_t>& inputShape, std::size_t maxBatch)
		: impl(std::make_unique<Impl>(layers, inputShape, maxBatch)) {
	}

	GpuConvNet::~GpuConvNet() = default;
	GpuConvNet::GpuConvNet(GpuConvNet&&) noexcept = default;
	GpuConvNet& GpuConvNet::operator=(GpuConvNet&&) noexcept = default;

	std::size_t GpuConvNet::maxBatch() const {
		return impl->batchCapacity;
	}

	std::size_t GpuConvNet::classCount() const {
		return impl->classes;
	}

	void GpuConvNet::uploadParameters(const std::vector<std::vector<float>>& values, std::size_t stepsTaken) {
		std::vector<DeviceParameter*> parameters = impl->allParameters();
		if (values.size() != parameters.size()) {
			throw InvalidSizeError("GpuConvNet::uploadParameters: wrong number of parameters!");
		}
		for (std::size_t p = 0; p < parameters.size(); p++) {
			if (values[p].size() != parameters[p]->count) {
				throw InvalidSizeError("GpuConvNet::uploadParameters: parameter " + std::to_string(p) + " has the wrong size!");
			}
		}
		for (std::size_t p = 0; p < parameters.size(); p++) {
			parameters[p]->value.upload(values[p].data(), values[p].size());
			parameters[p]->grad.zero();
			parameters[p]->resetOptimizerState();
		}
		impl->stepCount = stepsTaken;
	}

	void GpuConvNet::downloadParameters(std::vector<std::vector<float>>& values) const {
		values.clear();
		for (DeviceParameter* parameter : impl->allParameters()) {
			values.push_back(parameter->value.toVector());
		}
	}

	void GpuConvNet::downloadGradients(std::vector<std::vector<float>>& gradients) const {
		gradients.clear();
		for (DeviceParameter* parameter : impl->allParameters()) {
			gradients.push_back(parameter->grad.toVector());
		}
	}

	void GpuConvNet::uploadTrainingSet(const std::vector<float>& pixels, const std::vector<std::uint8_t>& labels) {
		const std::size_t sampleSize = impl->bufferSampleSizes[0];
		if (labels.empty() || pixels.size() != labels.size() * sampleSize) {
			throw InvalidSizeError("GpuConvNet::uploadTrainingSet: pixels and labels do not match the input shape!");
		}
		std::vector<int> labelValues(labels.size());
		for (std::size_t i = 0; i < labels.size(); i++) {
			if (labels[i] >= impl->classes) {
				throw InvalidParameterError("GpuConvNet::uploadTrainingSet: a label is not below the class count!");
			}
			labelValues[i] = labels[i];
		}
		impl->setPixels.allocate(pixels.size());
		impl->setPixels.upload(pixels.data(), pixels.size());
		impl->setLabels.allocate(labelValues.size());
		impl->setLabels.upload(labelValues.data(), labelValues.size());
		impl->setSize = labels.size();
	}

	std::size_t GpuConvNet::trainingSetSize() const {
		return impl->setSize;
	}

	float GpuConvNet::computeGradients(const std::vector<float>& images, const std::vector<std::size_t>& labels) {
		const std::size_t batch = labels.size();
		impl->requireBatch(batch);
		if (images.size() != batch * impl->bufferSampleSizes[0]) {
			throw InvalidSizeError("GpuConvNet::computeGradients: images do not match the labels and input shape!");
		}
		std::vector<int> labelValues(batch);
		for (std::size_t n = 0; n < batch; n++) {
			if (labels[n] >= impl->classes) {
				throw InvalidParameterError("GpuConvNet::computeGradients: label " + std::to_string(labels[n])
					+ " is not below the class count!");
			}
			labelValues[n] = static_cast<int>(labels[n]);
		}
		impl->activations[0].upload(images.data(), images.size());
		impl->batchLabels.upload(labelValues.data(), batch);
		impl->forward(batch);
		return impl->lossAndBackward(batch);
	}

	float GpuConvNet::computeGradientsResident(const std::vector<std::uint32_t>& indices) {
		if (impl->setSize == 0) {
			throw InvalidSizeError("GpuConvNet::computeGradientsResident: no training set was uploaded!");
		}
		const std::size_t batch = indices.size();
		impl->requireBatch(batch);
		for (std::uint32_t index : indices) {
			if (index >= impl->setSize) {
				throw InvalidParameterError("GpuConvNet::computeGradientsResident: image index out of range!");
			}
		}
		impl->batchIndices.upload(indices.data(), batch);
		const std::size_t sampleSize = impl->bufferSampleSizes[0];
		const std::size_t total = batch * sampleSize;
		kernels::gatherBatch<<<blocksFor(total), blockSize>>>(total, sampleSize, impl->batchIndices.get(),
			impl->setPixels.get(), impl->setLabels.get(), impl->activations[0].get(), impl->batchLabels.get());
		check(cudaGetLastError(), "gatherBatch kernel launch");
		impl->forward(batch);
		return impl->lossAndBackward(batch);
	}

	void GpuConvNet::update(const OptimizerSettings& optimizer) {
		optimizer.validate();
		impl->stepCount++;
		const float lr = static_cast<float>(optimizer.learningRate);

		double correction1 = 1.0;
		double correction2 = 1.0;
		if (optimizer.kind == OptimizerKind::Adam) {
			optimizer.adamBiasCorrections(impl->stepCount, correction1, correction2);
		}

		for (DeviceParameter* parameter : impl->allParameters()) {
			const std::size_t count = parameter->count;
			switch (optimizer.kind) {
			case OptimizerKind::SGD:
				kernels::sgdUpdate<<<blocksFor(count), blockSize>>>(count, parameter->value.get(), parameter->grad.get(), lr);
				break;
			case OptimizerKind::Momentum:
				parameter->ensureMoments(false);
				kernels::momentumUpdate<<<blocksFor(count), blockSize>>>(count, parameter->value.get(),
					parameter->grad.get(), parameter->firstMoment.get(), lr, static_cast<float>(optimizer.momentum));
				break;
			case OptimizerKind::Adam:
				parameter->ensureMoments(true);
				kernels::adamUpdate<<<blocksFor(count), blockSize>>>(count, parameter->value.get(), parameter->grad.get(),
					parameter->firstMoment.get(), parameter->secondMoment.get(), lr,
					static_cast<float>(optimizer.beta1), static_cast<float>(optimizer.beta2), static_cast<float>(optimizer.epsilon),
					static_cast<float>(correction1), static_cast<float>(correction2));
				break;
			}
			check(cudaGetLastError(), "optimizer kernel launch");
		}
	}

	std::size_t GpuConvNet::steps() const {
		return impl->stepCount;
	}

	void GpuConvNet::logits(const std::vector<float>& images, std::size_t count, std::vector<float>& result) {
		const std::size_t sampleSize = impl->bufferSampleSizes[0];
		if (count == 0 || images.size() != count * sampleSize) {
			throw InvalidSizeError("GpuConvNet::logits: images do not match the count and input shape!");
		}
		result.resize(count * impl->classes);
		const std::size_t logitsBuffer = impl->layers.back().outputBuffer;
		for (std::size_t start = 0; start < count; start += impl->batchCapacity) {
			const std::size_t batch = std::min(impl->batchCapacity, count - start);
			impl->activations[0].upload(images.data() + start * sampleSize, batch * sampleSize);
			impl->forward(batch);
			impl->activations[logitsBuffer].download(result.data() + start * impl->classes, batch * impl->classes);
		}
	}

} // namespace cuda
