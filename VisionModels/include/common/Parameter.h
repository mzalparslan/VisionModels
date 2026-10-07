#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "Optimizer.h"
#include "Tensor.h"
#include "Validation.h"

/**
 * @brief Random number engine used to draw initial weights.
 *
 * There is no global engine. A model is built from its own seed and passes
 * the engine to every layer, so a given seed always reproduces the same
 * weights (with a given standard library: std::normal_distribution is not
 * specified bit for bit, so MSVC and GCC draw different values).
 */
using RandomEngine = std::mt19937;

/**
 * @brief Learnable parameter: weights or biases, their gradient, and the
 * optimizer state the update needs.
 */
template <typename T>
class Parameter {
public:
	// Weights or biases.
	Tensor<T> value;
	// Gradient of the loss (dL/dvalue), accumulated by backward passes.
	Tensor<T> grad;
	// Momentum's velocity, or Adam's first moment m. Allocated on first use.
	Tensor<T> firstMoment;
	// Adam's second moment v. Allocated on first use.
	Tensor<T> secondMoment;

	/**
	 * @brief Weights drawn from N(0, stddev^2), gradients 0, optimizer state cleared.
	 *
	 * Random weights break symmetry: if every filter started equal, every
	 * filter would get the same gradient and they would stay equal forever.
	 *
	 * @throws InvalidParameterSizeError If shape is empty or has a zero dimension.
	 * @throws InvalidParameterError, NaNError, NonFiniteError If stddev is negative or not finite.
	 */
	void initNormal(const std::vector<std::size_t>& shape, T stddev, RandomEngine& rng) {
		validation::requireValidShape(shape, "Parameter shape");
		validation::requireNonNegativeFinite(stddev, "Parameter initialization stddev");

		value = Tensor<T>(shape);
		grad = Tensor<T>(shape, T(0));
		firstMoment = Tensor<T>();
		secondMoment = Tensor<T>();

		std::normal_distribution<T> dist(T(0), T(1));
		for (auto& x : value.data) {
			x = dist(rng) * stddev;
		}
	}

	/**
	 * @brief Every weight set to a constant (biases start at 0), gradients 0.
	 *
	 * @throws InvalidParameterSizeError If shape is empty or has a zero dimension.
	 * @throws NaNError, NonFiniteError If fill is not finite.
	 */
	void initConstant(const std::vector<std::size_t>& shape, T fill) {
		validation::requireValidShape(shape, "Parameter shape");
		validation::requireFinite(fill, "Parameter constant");

		value = Tensor<T>(shape, fill);
		grad = Tensor<T>(shape, T(0));
		firstMoment = Tensor<T>();
		secondMoment = Tensor<T>();
	}

	/**
	 * @brief Resets the gradient to 0. Backward passes accumulate (+=) into
	 * grad, so call this once per training step before them.
	 */
	void zeroGrad() {
		std::fill(grad.data.begin(), grad.data.end(), T(0));
	}

	/**
	 * @brief Clears optimizer state (momentum / Adam moments), e.g. after
	 * weights were replaced from outside.
	 */
	void resetOptimizerState() {
		firstMoment = Tensor<T>();
		secondMoment = Tensor<T>();
	}

	/**
	 * @brief One optimizer step on this parameter.
	 *
	 * @param settings Optimizer and hyper-parameters.
	 * @param step 1-based count of steps taken so far including this one
	 * (Adam's bias correction needs it; the others ignore it).
	 * @throws InvalidParameterError If settings are invalid or Adam's step is 0.
	 * @throws InvalidParameterSizeError If gradient and weights differ in size.
	 * @throws NaNError, NonFiniteError If a gradient or an updated weight is
	 * not finite: a bad value is reported before training carries on with it.
	 */
	void update(const OptimizerSettings& settings, std::size_t step) {
		settings.validate();
		validation::requirePositiveSize(value.size(), "Parameter size");
		if (grad.size() != value.size()) {
			throw InvalidParameterSizeError("Parameter gradient size differs from its weights!");
		}

		const T lr = static_cast<T>(settings.learningRate);
		switch (settings.kind) {
		case OptimizerKind::SGD:
			for (std::size_t i = 0; i < value.size(); i++) {
				validation::requireFinite(grad[i], "Gradient");
				value[i] -= lr * grad[i];
				validation::requireFinite(value[i], "Updated weight");
			}
			break;

		case OptimizerKind::Momentum: {
			ensureMoments(false);
			const T momentum = static_cast<T>(settings.momentum);
			for (std::size_t i = 0; i < value.size(); i++) {
				validation::requireFinite(grad[i], "Gradient");
				firstMoment[i] = momentum * firstMoment[i] + grad[i];
				value[i] -= lr * firstMoment[i];
				validation::requireFinite(value[i], "Updated weight");
			}
			break;
		}

		case OptimizerKind::Adam: {
			double correction1 = 0.0;
			double correction2 = 0.0;
			settings.adamBiasCorrections(step, correction1, correction2);
			ensureMoments(true);

			const T beta1 = static_cast<T>(settings.beta1);
			const T beta2 = static_cast<T>(settings.beta2);
			const T epsilon = static_cast<T>(settings.epsilon);
			const T biasCorrection1 = static_cast<T>(correction1);
			const T biasCorrection2 = static_cast<T>(correction2);

			for (std::size_t i = 0; i < value.size(); i++) {
				validation::requireFinite(grad[i], "Gradient");
				const T g = grad[i];
				// m_t = beta1 * m_{t-1} + (1 - beta1) * g
				firstMoment[i] = beta1 * firstMoment[i] + (T(1) - beta1) * g;
				// v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
				secondMoment[i] = beta2 * secondMoment[i] + (T(1) - beta2) * g * g;
				const T mHat = firstMoment[i] / biasCorrection1;
				const T vHat = secondMoment[i] / biasCorrection2;
				// w_t = w_{t-1} - lr * mHat / (sqrt(vHat) + epsilon)
				value[i] -= lr * mHat / (std::sqrt(vHat) + epsilon);
				validation::requireFinite(value[i], "Updated weight");
			}
			break;
		}
		}
	}

private:
	void ensureMoments(bool needSecond) {
		if (firstMoment.size() != value.size()) {
			firstMoment = Tensor<T>(value.shape, T(0));
		}
		if (needSecond && secondMoment.size() != value.size()) {
			secondMoment = Tensor<T>(value.shape, T(0));
		}
	}
};
