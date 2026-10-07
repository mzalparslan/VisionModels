//
// TestSupport.h
//
// Helpers shared by unit tests: deterministic tensors, finite-difference
// gradient checking, and a check that a layer's backward pass matches
// its forward pass.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include "Layer.h"
#include "Tensor.h"

namespace testsupport {

	/**
	 * @brief A tensor of the given shape filled with values from N(0, 1),
	 * reproducible from `seed`.
	 */
	template <typename T = double>
	Tensor<T> randomTensor(const std::vector<std::size_t>& shape, unsigned seed) {
		Tensor<T> tensor(shape);
		std::mt19937 rng(seed);
		std::normal_distribution<double> dist(0.0, 1.0);
		for (auto& x : tensor.data) {
			x = static_cast<T>(dist(rng));
		}
		return tensor;
	}

	/**
	 * @brief Builds a tensor from a shape and its values in order.
	 */
	inline Tensor<double> makeTensor(const std::vector<std::size_t>& shape, const std::vector<double>& values) {
		Tensor<double> tensor(shape);
		EXPECT_EQ(values.size(), tensor.size()) << "makeTensor: wrong number of values";
		for (std::size_t i = 0; i < tensor.size() && i < values.size(); i++) {
			tensor[i] = values[i];
		}
		return tensor;
	}

	/**
	 * @brief Compares shapes exactly and values within tolerance.
	 */
	template <typename T>
	::testing::AssertionResult tensorsNear(const Tensor<T>& actual, const Tensor<T>& expected, double tolerance) {
		if (actual.shape != expected.shape) {
			return ::testing::AssertionFailure() << "shape differs";
		}
		for (std::size_t i = 0; i < actual.size(); i++) {
			if (!(std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i])) <= tolerance)) {
				return ::testing::AssertionFailure() << "element " << i << ": " << actual[i] << " vs " << expected[i];
			}
		}
		return ::testing::AssertionSuccess();
	}

	/**
	 * @brief Sum of tensor * weights, element by element: a scalar "loss"
	 * whose gradient with respect to tensor is exactly weights.
	 */
	inline double weightedSum(const Tensor<double>& tensor, const Tensor<double>& weights) {
		double total = 0.0;
		for (std::size_t i = 0; i < tensor.size(); i++) {
			total += tensor[i] * weights[i];
		}
		return total;
	}

	/**
	 * @brief Central finite-difference derivative of lossFunction with respect
	 * to element `index` of `variable`, which is restored afterwards.
	 */
	inline double numericGradient(Tensor<double>& variable, std::size_t index,
		const std::function<double()>& lossFunction, double step = 1e-6) {
		const double original = variable[index];
		variable[index] = original + step;
		const double plusLoss = lossFunction();
		variable[index] = original - step;
		const double minusLoss = lossFunction();
		variable[index] = original;
		return (plusLoss - minusLoss) / (2.0 * step);
	}

	/**
	 * @brief Checks an analytic gradient against finite differences for every
	 * element of `variable` (relative tolerance, floored at 1).
	 */
	inline ::testing::AssertionResult gradientMatches(Tensor<double>& variable, const Tensor<double>& analytic,
		const std::function<double()>& lossFunction, double tolerance = 1e-5) {
		if (analytic.size() != variable.size()) {
			return ::testing::AssertionFailure() << "gradient has a different size";
		}
		for (std::size_t i = 0; i < variable.size(); i++) {
			const double numeric = numericGradient(variable, i, lossFunction);
			const double scale = std::max(1.0, std::max(std::fabs(numeric), std::fabs(analytic[i])));
			if (!(std::fabs(numeric - analytic[i]) <= tolerance * scale)) {
				return ::testing::AssertionFailure() << "element " << i << ": analytic " << analytic[i]
					<< " vs numeric " << numeric;
			}
		}
		return ::testing::AssertionSuccess();
	}

	/**
	 * @brief Full gradient check of a layer: with loss = sum(forward(x) * R),
	 * backward(R) must give dLoss/dx, and every parameter's grad must give
	 * dLoss/dparameter, all compared with finite differences.
	 */
	inline void expectLayerGradientsMatch(Layer<double>& layer, Tensor<double> input, unsigned seed,
		double tolerance = 1e-5) {
		const Tensor<double> firstOutput = layer.forward(input, 1);
		const Tensor<double> weights = randomTensor(firstOutput.shape, seed);

		for (Parameter<double>* parameter : layer.parameters()) {
			parameter->zeroGrad();
		}
		layer.forward(input, 1);
		const Tensor<double> gradInput = layer.backward(weights, 1);

		auto loss = [&]() { return weightedSum(layer.forward(input, 1), weights); };
		EXPECT_TRUE(gradientMatches(input, gradInput, loss, tolerance)) << "input gradient";

		std::size_t index = 0;
		for (Parameter<double>* parameter : layer.parameters()) {
			const Tensor<double> analytic = parameter->grad;
			EXPECT_TRUE(gradientMatches(parameter->value, analytic, loss, tolerance)) << "parameter " << index;
			index++;
		}
	}

} // namespace testsupport
