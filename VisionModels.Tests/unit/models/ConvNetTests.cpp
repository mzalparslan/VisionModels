#include "pch.h"

#include <numeric>

#include "DigitCnn.h"
#include "SyntheticDigits.h"
#include "TestSupport.h"

using testsupport::randomTensor;

namespace {

	// Every layer kind, small enough for a full finite-difference check.
	ConvNet<double> makeTinyNet() {
		RandomEngine rng(9);
		ConvNet<double> net({ 2, 6, 6 });
		net.addConv2D(3, 3, 1, 1, rng)
			.addReLU()
			.addMaxPool2D(2, 2)
			.addConv2D(4, 2, 1, 0, rng)
			.addReLU()
			.addFlatten()
			.addDense(5, rng)
			.addReLU()
			.addDense(3, rng);
		return net;
	}

}

TEST(ConvNetTests, BuildsShapesLayerByLayer) {
	ConvNet<float> net = makeDigitCnn<float>();
	EXPECT_EQ(net.inputShape(), (std::vector<std::size_t>{ 1, 28, 28 }));
	EXPECT_EQ(net.classCount(), 10u);
	EXPECT_EQ(net.layerCount(), 10u);
	// conv1 8*1*9+8, conv2 16*8*9+16, dense 64*784+64, dense 10*64+10.
	EXPECT_EQ(net.parameterCount(), 80u + 1168u + 50240u + 650u);
	EXPECT_NE(net.summary().find("Total parameters: 52138"), std::string::npos);
}

TEST(ConvNetTests, RejectsLayersThatDoNotFit) {
	RandomEngine rng(1);
	ConvNet<double> net({ 1, 4, 4 });
	EXPECT_THROW(net.addDense(3, rng), InvalidSizeError);
	EXPECT_THROW(net.addConv2D(2, 5, 1, 0, rng), InvalidSizeError);
	EXPECT_THROW(net.classCount(), InvalidSizeError);
	net.addFlatten();
	EXPECT_THROW(net.addConv2D(2, 3, 1, 0, rng), InvalidSizeError);
	EXPECT_THROW(ConvNet<double>({ 1, 4 }), InvalidSizeError);
}

TEST(ConvNetTests, RejectsImagesOfTheWrongShape) {
	ConvNet<double> net = makeTinyNet();
	EXPECT_THROW(net.forward(Tensor<double>({ 1, 2, 5, 6 })), InvalidSizeError);
	EXPECT_THROW(net.forward(Tensor<double>({ 1, 3, 6, 6 })), InvalidSizeError);
}

TEST(ConvNetTests, WholeNetworkGradientsMatchFiniteDifferences) {
	ConvNet<double> net = makeTinyNet();
	Tensor<double> images = randomTensor({ 3, 2, 6, 6 }, 91);
	const std::vector<std::size_t> labels = { 0, 2, 1 };

	net.computeGradients(images, labels);
	auto loss = [&]() { return SoftmaxCrossEntropy<double>::loss(net.forward(images), labels); };

	std::size_t index = 0;
	for (Parameter<double>* parameter : net.parameters()) {
		const Tensor<double> analytic = parameter->grad;
		EXPECT_TRUE(testsupport::gradientMatches(parameter->value, analytic, loss, 1e-5)) << "parameter " << index;
		index++;
	}
}

TEST(ConvNetTests, ParallelTrainingIsBitIdenticalToSequential) {
	ConvNet<float> sequential = makeDigitCnn<float>();
	ConvNet<float> parallel = makeDigitCnn<float>();
	const ImageDataset data = SyntheticDigits::generate(48, 3);
	std::vector<std::size_t> indices(16);

	for (std::size_t step = 0; step < 3; step++) {
		std::iota(indices.begin(), indices.end(), step * 16);
		const Tensor<float> images = data.batchImages<float>(indices);
		const std::vector<std::size_t> labels = data.batchLabels(indices);
		const float lossSequential = sequential.trainBatch(images, labels, OptimizerSettings::adam(1e-3), 1);
		const float lossParallel = parallel.trainBatch(images, labels, OptimizerSettings::adam(1e-3), 8);
		EXPECT_EQ(lossSequential, lossParallel);
	}

	const auto valuesSequential = sequential.parameterValues<float>();
	const auto valuesParallel = parallel.parameterValues<float>();
	EXPECT_EQ(valuesSequential, valuesParallel);
}

TEST(ConvNetTests, TrainingLowersTheLossAndLearnsSyntheticDigits) {
	ConvNet<float> net = makeDigitCnn<float>();
	const ImageDataset training = SyntheticDigits::generate(3000, 1);
	const ImageDataset test = SyntheticDigits::generate(300, 2);

	std::mt19937 shuffle(1);
	float firstLoss = 0.0f;
	float lastLoss = 0.0f;
	for (std::size_t epoch = 0; epoch < 3; epoch++) {
		const std::vector<std::size_t> order = training.shuffledIndices(shuffle);
		for (std::size_t start = 0; start < order.size(); start += 32) {
			const std::vector<std::size_t> batch(order.begin() + start, order.begin() + std::min(order.size(), start + 32));
			lastLoss = net.trainBatch(training.batchImages<float>(batch), training.batchLabels(batch),
				OptimizerSettings::adam(3e-3), 8);
			if (epoch == 0 && start == 0) {
				firstLoss = lastLoss;
			}
		}
	}
	EXPECT_LT(lastLoss, 0.5f * firstLoss);

	std::vector<std::size_t> all(test.size());
	std::iota(all.begin(), all.end(), std::size_t(0));
	const std::vector<std::size_t> predicted = net.predict(test.batchImages<float>(all), 8);
	std::size_t correct = 0;
	for (std::size_t i = 0; i < test.size(); i++) {
		correct += predicted[i] == test.labels[i] ? 1 : 0;
	}
	EXPECT_GT(static_cast<double>(correct) / static_cast<double>(test.size()), 0.9);
}

TEST(ConvNetTests, ParameterValuesRoundTrip) {
	ConvNet<double> source = makeTinyNet();
	ConvNet<double> target = makeTinyNet();
	for (Parameter<double>* parameter : source.parameters()) {
		for (double& x : parameter->value.data) {
			x += 0.25;
		}
	}
	target.setParameterValues(source.parameterValues<double>(), 17);
	EXPECT_EQ(target.parameterValues<double>(), source.parameterValues<double>());
	EXPECT_EQ(target.steps(), 17u);

	std::vector<std::vector<double>> wrong = source.parameterValues<double>();
	wrong.pop_back();
	EXPECT_THROW(target.setParameterValues(wrong, 0), InvalidSizeError);
}

TEST(ConvNetTests, SpecsDescribeEveryLayer) {
	ConvNet<float> net = makeDigitCnn<float>();
	const std::vector<LayerSpec> specs = net.specs();
	ASSERT_EQ(specs.size(), 10u);
	EXPECT_EQ(specs[0].kind, LayerKind::Conv2D);
	EXPECT_EQ(specs[0].outChannels, 8u);
	EXPECT_EQ(specs[2].kind, LayerKind::MaxPool2D);
	EXPECT_EQ(specs[6].kind, LayerKind::Flatten);
	EXPECT_EQ(specs[7].kind, LayerKind::Dense);
	EXPECT_EQ(specs[7].inFeatures, 784u);
	EXPECT_EQ(specs[9].outFeatures, 10u);
}
