#pragma once

#include <vector>
#include <limits>
#include <cstddef>
#include <stdexcept>

#include "Exceptions.h"

/**
 * @brief Flat 1D vector representing a multi-dimensional tensor.
 *
 * Exm: A 2x3 matrix [[1, 2, 3], [4, 5, 6]] is
 * stored as [1,2,3,4,5,6] with a Shape {2, 3}
 *
 * Images use NCHW layout, as in PyTorch and cuDNN's default: a batch of N
 * images with C channels, H rows and W columns has shape {N, C, H, W}, and
 * element (n, c, h, w) is at ((n * C + c) * H + h) * W + w. One row of pixels
 * is contiguous, so a convolution window slides along memory.
 */
template <typename T>
class Tensor {
public:
	std::vector<T> data;
	std::vector<std::size_t> shape;

	Tensor() {}

	/**
	 * @throws InvalidSizeError In case size exceeds max limit.
	 */
	Tensor(const std::vector<std::size_t>& shape, T value = T(0))
		: shape(shape)
	{
		data.assign(elementCount(shape), value);
	}

	/**
	 * @brief Number of elements a tensor of this shape holds.
	 *
	 * @throws InvalidSizeError In case size exceeds max limit.
	 */
	static std::size_t elementCount(const std::vector<std::size_t>& shape) {
		std::size_t rawSize = 1;

		for (auto shapeSize : shape) {
			// Check overflow.
			if ((shapeSize != 0) &&
				(std::numeric_limits<std::size_t>::max() / shapeSize) < rawSize) {
				throw InvalidSizeError("Overflow for tensor size!");
			}

			rawSize *= shapeSize;
		}
		return rawSize;
	}

	/**
	 * @brief Flat, rank-agnostic element access. Caller is responsible for
	 * computing correct flat offset.
	 */
	T& operator[](std::size_t i) { return data[i]; }
	const T& operator[](std::size_t i) const { return data[i]; }

	/**
	 * @brief Get total size of tensor.
	 */
	std::size_t size() const { return data.size(); }

	/**
	 * @brief Number of dimensions.
	 */
	std::size_t rank() const { return shape.size(); }

	/**
	 * @brief Changes the shape without touching the data (e.g. [N, C, H, W]
	 * to [N, C * H * W]).
	 *
	 * @throws InvalidSizeError If the new shape holds a different number of elements.
	 */
	void reshape(const std::vector<std::size_t>& newShape) {
		if (elementCount(newShape) != data.size()) {
			throw InvalidSizeError("Tensor reshape must keep the number of elements!");
		}
		shape = newShape;
	}
};
