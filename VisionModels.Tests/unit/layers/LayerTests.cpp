#include "pch.h"

#include "Dense.h"
#include "Flatten.h"
#include "MaxPool2D.h"
#include "ReLU.h"
#include "TestSupport.h"

using testsupport::makeTensor;
using testsupport::randomTensor;
using testsupport::tensorsNear;

// ---------------------------------------------------------------- MaxPool2D

TEST(MaxPool2DTests, ForwardKeepsTheLargestOfEachWindow) {
	MaxPool2D<double> pool(2, 2);
	const Tensor<double> input = makeTensor({ 1, 1, 4, 4 }, {
		1, 3, 2, 0,
		4, 2, 1, 5,
		0, 0, 7, 1,
		9, 1, 2, 2 });

	EXPECT_TRUE(tensorsNear(pool.forward(input, 1), makeTensor({ 1, 1, 2, 2 }, { 4, 5, 9, 7 }), 0.0));
}

TEST(MaxPool2DTests, BackwardSendsTheGradientToTheWinnerOnly) {
	MaxPool2D<double> pool(2, 2);
	const Tensor<double> input = makeTensor({ 1, 1, 2, 4 }, {
		1, 3, 2, 0,
		4, 2, 1, 5 });
	pool.forward(input, 1);

	const Tensor<double> gradInput = pool.backward(makeTensor({ 1, 1, 1, 2 }, { 10, 20 }), 1);

	EXPECT_TRUE(tensorsNear(gradInput, makeTensor({ 1, 1, 2, 4 }, {
		0, 0, 0, 0,
		10, 0, 0, 20 }), 0.0));
}

TEST(MaxPool2DTests, GradientsMatchFiniteDifferences) {
	MaxPool2D<double> pool(2, 2);
	testsupport::expectLayerGradientsMatch(pool, randomTensor({ 2, 3, 6, 5 }, 41), 42);
}

TEST(MaxPool2DTests, OverlappingWindowsAddTheirGradients) {
	MaxPool2D<double> pool(3, 1);
	testsupport::expectLayerGradientsMatch(pool, randomTensor({ 1, 2, 5, 5 }, 43), 44);
}

TEST(MaxPool2DTests, ParallelIsBitIdenticalToSequential) {
	MaxPool2D<double> sequential(2, 2);
	MaxPool2D<double> parallel(2, 2);
	const Tensor<double> input = randomTensor({ 6, 4, 8, 8 }, 45);
	const Tensor<double> gradOutput = randomTensor({ 6, 4, 4, 4 }, 46);
	EXPECT_EQ(sequential.forward(input, 1).data, parallel.forward(input, 8).data);
	EXPECT_EQ(sequential.backward(gradOutput, 1).data, parallel.backward(gradOutput, 8).data);
}

// --------------------------------------------------------------------- ReLU

TEST(ReLUTests, ForwardAndBackward) {
	ReLU<double> relu;
	const Tensor<double> input = makeTensor({ 1, 4 }, { -2, -0.5, 0.5, 3 });

	EXPECT_TRUE(tensorsNear(relu.forward(input, 1), makeTensor({ 1, 4 }, { 0, 0, 0.5, 3 }), 0.0));
	EXPECT_TRUE(tensorsNear(relu.backward(makeTensor({ 1, 4 }, { 1, 2, 3, 4 }), 1),
		makeTensor({ 1, 4 }, { 0, 0, 3, 4 }), 0.0));
}

TEST(ReLUTests, GradientsMatchFiniteDifferences) {
	ReLU<double> relu;
	testsupport::expectLayerGradientsMatch(relu, randomTensor({ 2, 3, 4, 4 }, 51), 52);
}

// ------------------------------------------------------------------ Flatten

TEST(FlattenTests, ReshapesAndRestores) {
	Flatten<double> flatten;
	const Tensor<double> input = randomTensor({ 2, 3, 4, 5 }, 61);

	const Tensor<double> output = flatten.forward(input, 1);
	EXPECT_EQ(output.shape, (std::vector<std::size_t>{ 2, 60 }));
	EXPECT_EQ(output.data, input.data);
	EXPECT_EQ(flatten.outputShape({ 3, 4, 5 }), (std::vector<std::size_t>{ 60 }));

	const Tensor<double> back = flatten.backward(output, 1);
	EXPECT_EQ(back.shape, input.shape);
}

// -------------------------------------------------------------------- Dense

TEST(DenseTests, ForwardIsWeightedSumPlusBias) {
	RandomEngine rng(1);
	Dense<double> dense(3, 2, rng);
	dense.weights.value = makeTensor({ 2, 3 }, { 1, 2, 3, -1, 0, 1 });
	dense.bias.value = makeTensor({ 2 }, { 0.5, -0.5 });

	const Tensor<double> output = dense.forward(makeTensor({ 2, 3 }, { 1, 1, 1, 2, 0, -1 }), 1);

	EXPECT_TRUE(tensorsNear(output, makeTensor({ 2, 2 }, { 6.5, -0.5, -0.5, -3.5 }), 1e-12));
}

TEST(DenseTests, GradientsMatchFiniteDifferences) {
	RandomEngine rng(2);
	Dense<double> dense(7, 4, rng);
	testsupport::expectLayerGradientsMatch(dense, randomTensor({ 3, 7 }, 71), 72);
}

TEST(DenseTests, ParallelIsBitIdenticalToSequential) {
	RandomEngine rngA(3);
	RandomEngine rngB(3);
	Dense<double> sequential(50, 20, rngA);
	Dense<double> parallel(50, 20, rngB);
	const Tensor<double> input = randomTensor({ 16, 50 }, 73);
	const Tensor<double> gradOutput = randomTensor({ 16, 20 }, 74);

	EXPECT_EQ(sequential.forward(input, 1).data, parallel.forward(input, 8).data);
	EXPECT_EQ(sequential.backward(gradOutput, 1).data, parallel.backward(gradOutput, 8).data);
	EXPECT_EQ(sequential.weights.grad.data, parallel.weights.grad.data);
	EXPECT_EQ(sequential.bias.grad.data, parallel.bias.grad.data);
}

TEST(DenseTests, RejectsWrongWidth) {
	RandomEngine rng(4);
	Dense<double> dense(3, 2, rng);
	EXPECT_THROW(dense.forward(Tensor<double>({ 1, 4 }), 1), InvalidSizeError);
	EXPECT_THROW(dense.outputShape({ 1, 3 }), InvalidSizeError);
}
