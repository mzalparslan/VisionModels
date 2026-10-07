#pragma once

#include <cstddef>
#include <string>

/**
 * @brief Kinds of layers a ConvNet is built from.
 */
enum class LayerKind {
	Conv2D,
	ReLU,
	MaxPool2D,
	Flatten,
	Dense
};

/**
 * @brief Plain description of one layer: its kind and sizes, no weights.
 *
 * It lets code that cannot include the templated CPU layers (the CUDA project,
 * compiled by nvcc) rebuild the same network: ConvNet::specs() lists the
 * layers, and cuda::GpuConvNet is constructed from that list. Fields that do
 * not apply to a kind are 0.
 */
class LayerSpec {
public:
	LayerKind kind = LayerKind::ReLU;

	// Conv2D.
	std::size_t inChannels = 0;
	std::size_t outChannels = 0;
	// Conv2D and MaxPool2D: square window size and step.
	std::size_t kernelSize = 0;
	std::size_t stride = 0;
	// Conv2D: zeros added around every side of the input.
	std::size_t padding = 0;

	// Dense.
	std::size_t inFeatures = 0;
	std::size_t outFeatures = 0;

	static LayerSpec conv2D(std::size_t inChannels, std::size_t outChannels, std::size_t kernelSize,
		std::size_t stride, std::size_t padding) {
		LayerSpec spec;
		spec.kind = LayerKind::Conv2D;
		spec.inChannels = inChannels;
		spec.outChannels = outChannels;
		spec.kernelSize = kernelSize;
		spec.stride = stride;
		spec.padding = padding;
		return spec;
	}

	static LayerSpec relu() {
		LayerSpec spec;
		spec.kind = LayerKind::ReLU;
		return spec;
	}

	static LayerSpec maxPool2D(std::size_t kernelSize, std::size_t stride) {
		LayerSpec spec;
		spec.kind = LayerKind::MaxPool2D;
		spec.kernelSize = kernelSize;
		spec.stride = stride;
		return spec;
	}

	static LayerSpec flatten() {
		LayerSpec spec;
		spec.kind = LayerKind::Flatten;
		return spec;
	}

	static LayerSpec dense(std::size_t inFeatures, std::size_t outFeatures) {
		LayerSpec spec;
		spec.kind = LayerKind::Dense;
		spec.inFeatures = inFeatures;
		spec.outFeatures = outFeatures;
		return spec;
	}

	/**
	 * @brief Whether this kind of layer has weights (Conv2D and Dense: a
	 * weight tensor and a bias, in that order).
	 */
	bool hasParameters() const {
		return kind == LayerKind::Conv2D || kind == LayerKind::Dense;
	}
};

/**
 * @brief Output size of a sliding window along one dimension:
 * floor((input + 2 * padding - kernel) / stride) + 1, or 0 when the window
 * does not fit at all.
 */
inline std::size_t slidingWindowOutputSize(std::size_t input, std::size_t kernel,
	std::size_t stride, std::size_t padding) {
	if (stride == 0 || kernel == 0 || input + 2 * padding < kernel) {
		return 0;
	}
	return (input + 2 * padding - kernel) / stride + 1;
}
