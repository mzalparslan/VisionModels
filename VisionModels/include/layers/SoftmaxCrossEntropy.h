#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "Tensor.h"
#include "Validation.h"

/**
 * @brief Softmax followed by cross-entropy loss, the standard classifier
 * loss, computed together.
 *
 * softmax turns a sample's class scores (logits) z into probabilities
 * p_k = exp(z_k) / sum_j exp(z_j). Cross-entropy is -log p_label: near 0 when
 * the right class gets probability near 1, large when it gets little.
 *
 * Doing both in one step has two benefits:
 *  - stability: subtracting the largest logit before exp() keeps it from
 *    overflowing, and log(sum exp) is computed without ever forming a tiny p;
 *  - a simple gradient: dL/dz_k = p_k - [k == label].
 *
 * The loss of a batch is the mean over its samples, so gradients are divided
 * by N and the learning rate does not depend on the batch size.
 */
template <typename T>
class SoftmaxCrossEntropy {
public:
	/**
	 * @brief Mean loss of a batch and its gradient with respect to the logits.
	 *
	 * @param logits [N, classes] scores.
	 * @param labels N class indices.
	 * @param gradLogits Result [N, classes].
	 * @throws InvalidSizeError If logits is not a non-empty matrix or labels has the wrong length.
	 * @throws InvalidParameterError If a label is not below the class count.
	 * @throws NaNError, NonFiniteError If the loss is not finite.
	 */
	static T lossAndGradient(const Tensor<T>& logits, const std::vector<std::size_t>& labels, Tensor<T>& gradLogits) {
		validation::requireMatrix(logits, "Logits");
		const std::size_t batch = logits.shape[0];
		const std::size_t classes = logits.shape[1];
		validation::requireSameSize(labels.size(), batch, "Labels");

		gradLogits = Tensor<T>(logits.shape);
		const T scale = T(1) / static_cast<T>(batch);
		T total = T(0);

		for (std::size_t n = 0; n < batch; n++) {
			validation::requireBelow(labels[n], classes, "Label");
			const T* z = &logits[n * classes];
			T* dz = &gradLogits[n * classes];

			const T maxLogit = *std::max_element(z, z + classes);
			T sumExp = T(0);
			for (std::size_t k = 0; k < classes; k++) {
				dz[k] = std::exp(z[k] - maxLogit);
				sumExp += dz[k];
			}
			// -log p_label = log(sum exp(z - max)) - (z_label - max)
			total += std::log(sumExp) - (z[labels[n]] - maxLogit);

			for (std::size_t k = 0; k < classes; k++) {
				dz[k] = (dz[k] / sumExp - (k == labels[n] ? T(1) : T(0))) * scale;
			}
		}

		const T loss = total * scale;
		validation::requireFinite(loss, "Cross-entropy loss");
		return loss;
	}

	/**
	 * @brief Mean loss only, without the gradient.
	 */
	static T loss(const Tensor<T>& logits, const std::vector<std::size_t>& labels) {
		Tensor<T> unused;
		return lossAndGradient(logits, labels, unused);
	}

	/**
	 * @brief Row-wise softmax of [N, classes] logits.
	 *
	 * @throws InvalidSizeError If logits is not a non-empty matrix.
	 */
	static Tensor<T> softmax(const Tensor<T>& logits) {
		validation::requireMatrix(logits, "Logits");
		const std::size_t batch = logits.shape[0];
		const std::size_t classes = logits.shape[1];
		Tensor<T> probabilities(logits.shape);

		for (std::size_t n = 0; n < batch; n++) {
			const T* z = &logits[n * classes];
			T* p = &probabilities[n * classes];
			const T maxLogit = *std::max_element(z, z + classes);
			T sumExp = T(0);
			for (std::size_t k = 0; k < classes; k++) {
				p[k] = std::exp(z[k] - maxLogit);
				sumExp += p[k];
			}
			for (std::size_t k = 0; k < classes; k++) {
				p[k] /= sumExp;
			}
		}
		return probabilities;
	}
};
