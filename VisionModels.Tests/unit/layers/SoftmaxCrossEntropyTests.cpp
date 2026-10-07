#include "pch.h"

#include <cmath>

#include "SoftmaxCrossEntropy.h"
#include "TestSupport.h"

using testsupport::makeTensor;
using testsupport::randomTensor;

TEST(SoftmaxCrossEntropyTests, EqualScoresGiveLogOfClassCount) {
	const Tensor<double> logits({ 2, 10 }, 3.0);
	EXPECT_NEAR(SoftmaxCrossEntropy<double>::loss(logits, { 0, 7 }), std::log(10.0), 1e-12);
}

TEST(SoftmaxCrossEntropyTests, ConfidentRightAnswerHasNearZeroLoss) {
	const Tensor<double> logits = makeTensor({ 1, 3 }, { 20, 0, 0 });
	EXPECT_LT(SoftmaxCrossEntropy<double>::loss(logits, { 0 }), 1e-8);
	EXPECT_GT(SoftmaxCrossEntropy<double>::loss(logits, { 1 }), 19.0);
}

TEST(SoftmaxCrossEntropyTests, GradientIsProbabilityMinusOneHotOverBatch) {
	const Tensor<double> logits = randomTensor({ 4, 5 }, 81);
	const std::vector<std::size_t> labels = { 0, 4, 2, 2 };
	Tensor<double> gradient;
	SoftmaxCrossEntropy<double>::lossAndGradient(logits, labels, gradient);

	const Tensor<double> probabilities = SoftmaxCrossEntropy<double>::softmax(logits);
	for (std::size_t n = 0; n < 4; n++) {
		for (std::size_t k = 0; k < 5; k++) {
			const double expected = (probabilities[n * 5 + k] - (k == labels[n] ? 1.0 : 0.0)) / 4.0;
			EXPECT_NEAR(gradient[n * 5 + k], expected, 1e-12);
		}
	}

	Tensor<double> variable = logits;
	auto loss = [&]() { return SoftmaxCrossEntropy<double>::loss(variable, labels); };
	EXPECT_TRUE(testsupport::gradientMatches(variable, gradient, loss));
}

TEST(SoftmaxCrossEntropyTests, HugeLogitsDoNotOverflow) {
	const Tensor<double> logits = makeTensor({ 1, 3 }, { 1000, 999, -1000 });
	Tensor<double> gradient;
	const double loss = SoftmaxCrossEntropy<double>::lossAndGradient(logits, { 1 }, gradient);
	EXPECT_NEAR(loss, std::log(1.0 + std::exp(1.0)) , 1e-9);
	for (double g : gradient.data) {
		EXPECT_TRUE(std::isfinite(g));
	}
}

TEST(SoftmaxCrossEntropyTests, RejectsBadLabels) {
	const Tensor<double> logits({ 2, 3 }, 0.0);
	EXPECT_THROW(SoftmaxCrossEntropy<double>::loss(logits, { 0 }), InvalidSizeError);
	EXPECT_THROW(SoftmaxCrossEntropy<double>::loss(logits, { 0, 3 }), InvalidParameterError);
}
