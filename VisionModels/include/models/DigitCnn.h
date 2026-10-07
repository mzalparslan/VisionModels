#pragma once

#include <cstddef>
#include <cstdint>

#include "ConvNet.h"

/**
 * @brief Sizes of the digit-recognition CNN built by makeDigitCnn().
 */
class DigitCnnConfig {
public:
	// Filters in the first and second convolution.
	std::size_t conv1Channels = 8;
	std::size_t conv2Channels = 16;
	// Width of the hidden Dense layer.
	std::size_t hiddenUnits = 64;
	// Seed of the random initial weights.
	std::uint32_t seed = 42;
};

/**
 * @brief A small LeNet-style CNN for 28x28 grayscale digits (MNIST), 10 classes.
 *
 *   [1, 28, 28]
 *   Conv2D 3x3, padding 1 -> [8, 28, 28]    edges and strokes
 *   ReLU
 *   MaxPool2D 2x2        -> [8, 14, 14]
 *   Conv2D 3x3, padding 1 -> [16, 14, 14]   corners, curves, stroke combinations
 *   ReLU
 *   MaxPool2D 2x2        -> [16, 7, 7]
 *   Flatten              -> [784]
 *   Dense                -> [64]
 *   ReLU
 *   Dense                -> [10]            one score per digit
 *
 * About 52,000 parameters with the default config; it reaches about 98-99%
 * test accuracy on MNIST after 1-3 epochs with Adam. LeNet-5 (LeCun et al.,
 * 1998) is the classic original of this design.
 */
template <typename T>
ConvNet<T> makeDigitCnn(const DigitCnnConfig& config = DigitCnnConfig()) {
	RandomEngine rng(config.seed);
	ConvNet<T> net({ 1, 28, 28 });
	net.addConv2D(config.conv1Channels, 3, 1, 1, rng)
		.addReLU()
		.addMaxPool2D(2, 2)
		.addConv2D(config.conv2Channels, 3, 1, 1, rng)
		.addReLU()
		.addMaxPool2D(2, 2)
		.addFlatten()
		.addDense(config.hiddenUnits, rng)
		.addReLU()
		.addDense(10, rng);
	return net;
}
