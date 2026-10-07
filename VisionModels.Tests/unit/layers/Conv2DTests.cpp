#include "pch.h"

#include "Conv2D.h"
#include "TestSupport.h"

using testsupport::makeTensor;
using testsupport::randomTensor;
using testsupport::tensorsNear;

namespace {

	Conv2D<double> makeConv(std::size_t inChannels, std::size_t outChannels, std::size_t kernel,
		std::size_t stride, std::size_t padding) {
		RandomEngine rng(3);
		return Conv2D<double>(inChannels, outChannels, kernel, stride, padding, rng);
	}

}

TEST(Conv2DTests, ForwardComputesWindowSumsPlusBias) {
	Conv2D<double> conv = makeConv(1, 1, 2, 1, 0);
	conv.weights.value = makeTensor({ 1, 1, 2, 2 }, { 1, 2, 3, 4 });
	conv.bias.value = makeTensor({ 1 }, { 0.5 });
	const Tensor<double> input = makeTensor({ 1, 1, 3, 3 }, {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9 });

	const Tensor<double> output = conv.forward(input, 1);

	// Top-left window [1 2; 4 5]: 1*1 + 2*2 + 3*4 + 4*5 = 37, plus 0.5.
	EXPECT_TRUE(tensorsNear(output, makeTensor({ 1, 1, 2, 2 }, { 37.5, 47.5, 67.5, 77.5 }), 1e-12));
}

TEST(Conv2DTests, PaddingReadsZerosOutsideTheImage) {
	Conv2D<double> conv = makeConv(1, 1, 3, 1, 1);
	conv.weights.value = Tensor<double>({ 1, 1, 3, 3 }, 1.0);
	conv.bias.value = makeTensor({ 1 }, { 0.0 });
	const Tensor<double> input = Tensor<double>({ 1, 1, 2, 2 }, 1.0);

	const Tensor<double> output = conv.forward(input, 1);

	// Every 3x3 window centred on a 2x2 image of ones covers all 4 pixels.
	EXPECT_TRUE(tensorsNear(output, Tensor<double>({ 1, 1, 2, 2 }, 4.0), 1e-12));
}

TEST(Conv2DTests, OutputShapeFollowsStrideAndPadding) {
	Conv2D<double> conv = makeConv(3, 5, 3, 2, 1);
	EXPECT_EQ(conv.outputShape({ 3, 28, 28 }), (std::vector<std::size_t>{ 5, 14, 14 }));
	EXPECT_EQ(conv.outputShape({ 3, 7, 9 }), (std::vector<std::size_t>{ 5, 4, 5 }));
	EXPECT_THROW(conv.outputShape({ 2, 28, 28 }), InvalidSizeError);
	EXPECT_EQ(conv.forward(randomTensor({ 2, 3, 7, 9 }, 1), 1).shape, (std::vector<std::size_t>{ 2, 5, 4, 5 }));
}

TEST(Conv2DTests, GradientsMatchFiniteDifferences) {
	Conv2D<double> conv = makeConv(2, 3, 3, 1, 1);
	testsupport::expectLayerGradientsMatch(conv, randomTensor({ 2, 2, 5, 4 }, 11), 12);
}

TEST(Conv2DTests, GradientsMatchWithStrideAndNoPadding) {
	Conv2D<double> conv = makeConv(2, 2, 3, 2, 0);
	testsupport::expectLayerGradientsMatch(conv, randomTensor({ 2, 2, 7, 6 }, 21), 22);
}

TEST(Conv2DTests, ParallelIsBitIdenticalToSequential) {
	Conv2D<double> sequential = makeConv(4, 6, 3, 1, 1);
	Conv2D<double> parallel = makeConv(4, 6, 3, 1, 1);
	const Tensor<double> input = randomTensor({ 5, 4, 9, 9 }, 31);
	const Tensor<double> gradOutput = randomTensor({ 5, 6, 9, 9 }, 32);

	const Tensor<double> outSequential = sequential.forward(input, 1);
	const Tensor<double> outParallel = parallel.forward(input, 8);
	const Tensor<double> gradSequential = sequential.backward(gradOutput, 1);
	const Tensor<double> gradParallel = parallel.backward(gradOutput, 8);

	EXPECT_EQ(outSequential.data, outParallel.data);
	EXPECT_EQ(gradSequential.data, gradParallel.data);
	EXPECT_EQ(sequential.weights.grad.data, parallel.weights.grad.data);
	EXPECT_EQ(sequential.bias.grad.data, parallel.bias.grad.data);
}

TEST(Conv2DTests, RejectsBadInputAndOrder) {
	Conv2D<double> conv = makeConv(1, 2, 3, 1, 0);
	EXPECT_THROW(conv.backward(Tensor<double>({ 1, 2, 1, 1 }), 1), PipelineStateError);
	EXPECT_THROW(conv.forward(Tensor<double>({ 1, 1, 2, 2 }), 1), InvalidSizeError);
	EXPECT_THROW(conv.forward(Tensor<double>({ 1, 3, 5, 5 }), 1), InvalidSizeError);
	conv.forward(Tensor<double>({ 1, 1, 5, 5 }), 1);
	EXPECT_THROW(conv.backward(Tensor<double>({ 1, 2, 2, 2 }), 1), InvalidSizeError);

	RandomEngine rng(1);
	EXPECT_THROW(Conv2D<double>(0, 1, 3, 1, 0, rng), InvalidParameterSizeError);
	EXPECT_THROW(Conv2D<double>(1, 1, 3, 0, 0, rng), InvalidParameterSizeError);
}

TEST(Conv2DTests, HeInitializationHasTheExpectedSpread) {
	RandomEngine rng(5);
	Conv2D<double> conv(16, 32, 3, 1, 1, rng);
	double sumSquares = 0.0;
	for (double w : conv.weights.value.data) {
		sumSquares += w * w;
	}
	const double variance = sumSquares / static_cast<double>(conv.weights.value.size());
	EXPECT_NEAR(variance, 2.0 / (16.0 * 9.0), 0.1 * 2.0 / (16.0 * 9.0));
	for (double b : conv.bias.value.data) {
		EXPECT_EQ(b, 0.0);
	}
}
