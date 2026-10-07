#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Tensor.h"
#include "Validation.h"

/**
 * @brief Labeled images held in memory, all the same size.
 *
 * Pixels are float in [0, 1] (0 = black background, 1 = full ink for MNIST),
 * stored image after image in NCHW order, so image i starts at
 * i * sampleSize(). Kept as float whatever the model's type: it halves the
 * memory of a double copy, and it is what the GPU takes as is.
 */
class ImageDataset {
public:
	std::size_t channels = 1;
	std::size_t height = 28;
	std::size_t width = 28;
	// [count, channels, height, width]
	std::vector<float> pixels;
	// One class index per image.
	std::vector<std::uint8_t> labels;

	std::size_t size() const { return labels.size(); }
	std::size_t sampleSize() const { return channels * height * width; }
	std::vector<std::size_t> sampleShape() const { return { channels, height, width }; }

	/**
	 * @brief Largest label + 1 (10 for digits).
	 */
	std::size_t classCount() const {
		if (labels.empty()) {
			return 0;
		}
		return static_cast<std::size_t>(*std::max_element(labels.begin(), labels.end())) + 1;
	}

	/**
	 * @throws InvalidSizeError If pixels and labels do not describe the same number of images.
	 */
	void validate() const {
		validation::requirePositiveSize(sampleSize(), "Image size");
		validation::requireSameSize(pixels.size(), labels.size() * sampleSize(), "Dataset pixels");
	}

	/**
	 * @brief The images at `indices` as a [N, C, H, W] tensor of T.
	 * @throws InvalidParameterError If an index is out of range.
	 */
	template <typename T>
	Tensor<T> batchImages(const std::vector<std::size_t>& indices) const {
		Tensor<T> batch({ indices.size(), channels, height, width });
		const std::size_t stride = sampleSize();
		for (std::size_t b = 0; b < indices.size(); b++) {
			validation::requireBelow(indices[b], size(), "Image index");
			const float* source = &pixels[indices[b] * stride];
			std::transform(source, source + stride, batch.data.begin() + b * stride,
				[](float x) { return static_cast<T>(x); });
		}
		return batch;
	}

	/**
	 * @brief The labels at `indices`.
	 * @throws InvalidParameterError If an index is out of range.
	 */
	std::vector<std::size_t> batchLabels(const std::vector<std::size_t>& indices) const {
		std::vector<std::size_t> result(indices.size());
		for (std::size_t b = 0; b < indices.size(); b++) {
			validation::requireBelow(indices[b], size(), "Image index");
			result[b] = labels[indices[b]];
		}
		return result;
	}

	/**
	 * @brief The first `count` images (all of them if count is 0 or too large).
	 */
	ImageDataset head(std::size_t count) const {
		if (count == 0 || count >= size()) {
			return *this;
		}
		ImageDataset result;
		result.channels = channels;
		result.height = height;
		result.width = width;
		result.pixels.assign(pixels.begin(), pixels.begin() + count * sampleSize());
		result.labels.assign(labels.begin(), labels.begin() + count);
		return result;
	}

	/**
	 * @brief 0, 1, ..., size() - 1 in a random order, for one epoch.
	 */
	std::vector<std::size_t> shuffledIndices(std::mt19937& rng) const {
		std::vector<std::size_t> order(size());
		std::iota(order.begin(), order.end(), std::size_t(0));
		std::shuffle(order.begin(), order.end(), rng);
		return order;
	}

	/**
	 * @brief Image `index` as text art (" .:-=+*#%@" by brightness), one line
	 * per row, for looking at data and predictions in a console.
	 */
	std::string asciiArt(std::size_t index) const {
		validation::requireBelow(index, size(), "Image index");
		static const char ramp[] = " .:-=+*#%@";
		std::string text;
		const float* image = &pixels[index * sampleSize()];
		for (std::size_t h = 0; h < height; h++) {
			for (std::size_t w = 0; w < width; w++) {
				const float value = std::clamp(image[h * width + w], 0.0f, 1.0f);
				text += ramp[static_cast<std::size_t>(value * 9.0f + 0.5f)];
			}
			text += '\n';
		}
		return text;
	}
};
