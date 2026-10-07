#pragma once

#include <cmath>
#include <cstddef>
#include <string>

#include "Exceptions.h"

/**
 * @brief Weight-update algorithm applied by Parameter::update() on the CPU and
 * by cuda::GpuConvNet::update() on the GPU (same formulas in both).
 */
enum class OptimizerKind {
	// Plain gradient descent: w -= lr * g.
	SGD,
	// SGD with momentum (PyTorch form): v = momentum * v + g; w -= lr * v.
	// The velocity v keeps rolling in directions the gradient keeps pointing,
	// and damps directions where it flips sign from batch to batch.
	Momentum,
	// Adaptive Moment Estimation: a per-weight learning rate from running
	// averages of the gradient (m) and of its square (v).
	Adam
};

/**
 * @brief Name of an optimizer, for logging.
 */
inline const char* toString(OptimizerKind kind) {
	switch (kind) {
	case OptimizerKind::SGD:
		return "SGD";
	case OptimizerKind::Momentum:
		return "Momentum";
	case OptimizerKind::Adam:
		return "Adam";
	}
	return "Unknown";
}

/**
 * @brief Which optimizer to use and its hyper-parameters.
 *
 * Plain C++ with no Tensor dependency, so the CUDA project can include it too.
 * Values are double and are rounded to the model's type T when used.
 */
class OptimizerSettings {
public:
	OptimizerKind kind = OptimizerKind::Adam;
	double learningRate = 1e-3;
	// Momentum only.
	double momentum = 0.9;
	// Adam only.
	double beta1 = 0.9;
	double beta2 = 0.999;
	double epsilon = 1e-8;

	static OptimizerSettings sgd(double learningRate) {
		OptimizerSettings settings;
		settings.kind = OptimizerKind::SGD;
		settings.learningRate = learningRate;
		return settings;
	}

	static OptimizerSettings withMomentum(double learningRate, double momentum = 0.9) {
		OptimizerSettings settings;
		settings.kind = OptimizerKind::Momentum;
		settings.learningRate = learningRate;
		settings.momentum = momentum;
		return settings;
	}

	static OptimizerSettings adam(double learningRate = 1e-3) {
		OptimizerSettings settings;
		settings.kind = OptimizerKind::Adam;
		settings.learningRate = learningRate;
		return settings;
	}

	/**
	 * @throws InvalidParameterError If a hyper-parameter is out of range.
	 */
	void validate() const {
		if (!(learningRate > 0.0) || !std::isfinite(learningRate)) {
			throw InvalidParameterError("Learning rate must be a finite value greater than zero!");
		}
		if (kind == OptimizerKind::Momentum && !(momentum >= 0.0 && momentum < 1.0)) {
			throw InvalidParameterError("Momentum must be in [0, 1)!");
		}
		if (kind == OptimizerKind::Adam) {
			if (!(beta1 >= 0.0 && beta1 < 1.0) || !(beta2 >= 0.0 && beta2 < 1.0)) {
				throw InvalidParameterError("Adam beta1 and beta2 must be in [0, 1)!");
			}
			if (!(epsilon > 0.0) || !std::isfinite(epsilon)) {
				throw InvalidParameterError("Adam epsilon must be a finite value greater than zero!");
			}
		}
	}

	/**
	 * @brief Adam's bias corrections 1 - beta^step for a 1-based step.
	 *
	 * Early on m and v are averages over few gradients and start at zero, so
	 * they are too small; dividing by 1 - beta^step scales them back up.
	 *
	 * @throws InvalidParameterError If step is 0 (the correction would be 0).
	 */
	void adamBiasCorrections(std::size_t step, double& correction1, double& correction2) const {
		if (step == 0) {
			throw InvalidParameterError("Adam timestep must be >= 1!");
		}
		correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
		correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
	}
};
