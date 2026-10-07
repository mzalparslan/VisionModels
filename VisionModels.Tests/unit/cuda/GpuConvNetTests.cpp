#include "pch.h"

#include <cmath>
#include <numeric>
#include <iostream>
#include <sstream>

#include "CudaRuntime.h"
#include "DigitCnn.h"
#include "GpuConvNet.h"
#include "ImageClassifierTrainer.h"
#include "SyntheticDigits.h"

// These compare the GPU with the CPU model in float. Without a GPU (or in a
// build without the CUDA Toolkit) each test reports itself skipped and
// passes. (Google Test 1.8 has no GTEST_SKIP.)
#define SKIP_WITHOUT_GPU() \
	if (!cuda::isAvailable()) { \
		std::cout << "[  SKIPPED ] " << cuda::describeDevice() << "\n"; \
		return; \
	}

namespace {

	// Every layer kind, including a strided, padded convolution and
	// overlapping pooling windows, which DigitCnn does not have.
	ConvNet<float> makeVariedNet() {
		RandomEngine rng(4);
		ConvNet<float> net({ 2, 11, 9 });
		net.addConv2D(4, 3, 2, 1, rng)
			.addReLU()
			.addMaxPool2D(3, 1)
			.addConv2D(3, 2, 1, 0, rng)
			.addFlatten()
			.addDense(6, rng)
			.addReLU()
			.addDense(4, rng);
		return net;
	}

	std::vector<float> randomImages(std::size_t count, std::size_t size, unsigned seed) {
		std::mt19937 rng(seed);
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		std::vector<float> images(count * size);
		for (float& x : images) {
			x = dist(rng);
		}
		return images;
	}

	::testing::AssertionResult vectorsNear(const std::vector<float>& actual, const std::vector<float>& expected,
		float tolerance) {
		if (actual.size() != expected.size()) {
			return ::testing::AssertionFailure() << "size " << actual.size() << " vs " << expected.size();
		}
		for (std::size_t i = 0; i < actual.size(); i++) {
			const float scale = std::max(1.0f, std::fabs(expected[i]));
			if (!(std::fabs(actual[i] - expected[i]) <= tolerance * scale)) {
				return ::testing::AssertionFailure() << "element " << i << ": " << actual[i] << " vs " << expected[i];
			}
		}
		return ::testing::AssertionSuccess();
	}

	Tensor<float> toTensor(const std::vector<float>& values, const std::vector<std::size_t>& shape) {
		Tensor<float> tensor(shape);
		tensor.data = values;
		return tensor;
	}

}

TEST(GpuConvNetTests, DescribesTheDevice) {
	SKIP_WITHOUT_GPU();
	EXPECT_FALSE(cuda::deviceInfo().name.empty());
}

TEST(GpuConvNetTests, LogitsMatchTheCpu) {
	SKIP_WITHOUT_GPU();
	ConvNet<float> net = makeVariedNet();
	cuda::GpuConvNet gpu(net.specs(), net.inputShape(), 4);
	gpu.uploadParameters(net.parameterValues<float>());

	// 10 images with maxBatch 4: three batches, the last one partial.
	const std::vector<float> images = randomImages(10, 2 * 11 * 9, 1);
	std::vector<float> gpuLogits;
	gpu.logits(images, 10, gpuLogits);

	const Tensor<float> cpuLogits = net.forward(toTensor(images, { 10, 2, 11, 9 }));
	EXPECT_TRUE(vectorsNear(gpuLogits, cpuLogits.data, 1e-4f));
}

TEST(GpuConvNetTests, GradientsMatchTheCpu) {
	SKIP_WITHOUT_GPU();
	ConvNet<float> net = makeVariedNet();
	cuda::GpuConvNet gpu(net.specs(), net.inputShape(), 8);
	gpu.uploadParameters(net.parameterValues<float>());

	const std::vector<float> images = randomImages(5, 2 * 11 * 9, 2);
	const std::vector<std::size_t> labels = { 0, 3, 1, 1, 2 };
	const float gpuLoss = gpu.computeGradients(images, labels);
	const float cpuLoss = net.computeGradients(toTensor(images, { 5, 2, 11, 9 }), labels);
	EXPECT_NEAR(gpuLoss, cpuLoss, 1e-5f);

	std::vector<std::vector<float>> gpuGradients;
	gpu.downloadGradients(gpuGradients);
	const std::vector<Parameter<float>*> parameters = net.parameters();
	ASSERT_EQ(gpuGradients.size(), parameters.size());
	for (std::size_t p = 0; p < parameters.size(); p++) {
		EXPECT_TRUE(vectorsNear(gpuGradients[p], parameters[p]->grad.data, 1e-4f)) << "parameter " << p;
	}
}

TEST(GpuConvNetTests, OptimizerStepsMatchTheCpu) {
	SKIP_WITHOUT_GPU();
	for (const OptimizerSettings& optimizer : { OptimizerSettings::sgd(0.05), OptimizerSettings::withMomentum(0.05),
		OptimizerSettings::adam(1e-2) }) {
		ConvNet<float> net = makeVariedNet();
		cuda::GpuConvNet gpu(net.specs(), net.inputShape(), 8);
		gpu.uploadParameters(net.parameterValues<float>());
		const std::vector<float> images = randomImages(6, 2 * 11 * 9, 3);
		const std::vector<std::size_t> labels = { 0, 1, 2, 3, 0, 1 };

		for (int step = 0; step < 3; step++) {
			gpu.computeGradients(images, labels);
			gpu.update(optimizer);
			net.trainBatch(toTensor(images, { 6, 2, 11, 9 }), labels, optimizer);
		}

		std::vector<std::vector<float>> gpuValues;
		gpu.downloadParameters(gpuValues);
		const std::vector<std::vector<float>> cpuValues = net.parameterValues<float>();
		EXPECT_EQ(gpu.steps(), 3u);
		for (std::size_t p = 0; p < cpuValues.size(); p++) {
			EXPECT_TRUE(vectorsNear(gpuValues[p], cpuValues[p], 1e-4f)) << toString(optimizer.kind) << " parameter " << p;
		}
	}
}

TEST(GpuConvNetTests, ResidentBatchesMatchHostBatches) {
	SKIP_WITHOUT_GPU();
	ConvNet<float> net = makeDigitCnn<float>();
	const ImageDataset data = SyntheticDigits::generate(40, 5);
	cuda::GpuConvNet resident(net.specs(), net.inputShape(), 16);
	cuda::GpuConvNet host(net.specs(), net.inputShape(), 16);
	resident.uploadParameters(net.parameterValues<float>());
	host.uploadParameters(net.parameterValues<float>());
	resident.uploadTrainingSet(data.pixels, data.labels);
	EXPECT_EQ(resident.trainingSetSize(), 40u);

	const std::vector<std::size_t> indices = { 39, 3, 17, 0, 22 };
	const std::vector<std::uint32_t> gpuIndices = { 39, 3, 17, 0, 22 };
	const Tensor<float> images = data.batchImages<float>(indices);

	EXPECT_FLOAT_EQ(resident.computeGradientsResident(gpuIndices), host.computeGradients(images.data, data.batchLabels(indices)));
	std::vector<std::vector<float>> residentGradients;
	std::vector<std::vector<float>> hostGradients;
	resident.downloadGradients(residentGradients);
	host.downloadGradients(hostGradients);
	EXPECT_EQ(residentGradients, hostGradients);
}

TEST(GpuConvNetTests, RejectsBadArguments) {
	SKIP_WITHOUT_GPU();
	ConvNet<float> net = makeDigitCnn<float>();
	cuda::GpuConvNet gpu(net.specs(), net.inputShape(), 4);
	gpu.uploadParameters(net.parameterValues<float>());
	const std::vector<float> fiveImages(5 * 784, 0.0f);

	EXPECT_THROW(gpu.computeGradients(fiveImages, { 0, 1, 2, 3, 4 }), InvalidSizeError);
	EXPECT_THROW(gpu.computeGradients(std::vector<float>(784), { 10 }), InvalidParameterError);
	EXPECT_THROW(gpu.computeGradientsResident({ 0 }), InvalidSizeError);
	EXPECT_THROW(gpu.uploadParameters({ { 1.0f } }), InvalidSizeError);
	EXPECT_THROW(cuda::GpuConvNet(net.specs(), { 1, 20, 20 }, 4), InvalidSizeError);
	EXPECT_THROW(cuda::GpuConvNet(net.specs(), net.inputShape(), 0), InvalidSizeError);
}

TEST(GpuConvNetTests, TrainerLearnsSyntheticDigitsOnTheGpu) {
	SKIP_WITHOUT_GPU();
	const ImageDataset training = SyntheticDigits::generate(3000, 1);
	const ImageDataset test = SyntheticDigits::generate(500, 2);
	ConvNet<float> net = makeDigitCnn<float>();
	TrainingOptions options;
	options.epochs = 3;
	options.batchSize = 32;
	options.optimizer = OptimizerSettings::adam(3e-3);
	options.strategy = ExecutionStrategy::Cuda;
	options.logEverySteps = 0;
	std::ostringstream log;
	ImageClassifierTrainer<float> trainer(net, options, log);

	const std::vector<EpochReport> reports = trainer.train(training, &test);
	EXPECT_GT(reports[2].validationAccuracy, 0.9);

	// The weights came back to the CPU model: it agrees with the GPU.
	TrainingOptions cpuOptions = options;
	cpuOptions.strategy = ExecutionStrategy::Sequential;
	ImageClassifierTrainer<float> cpuTrainer(net, cpuOptions, log);
	EXPECT_NEAR(cpuTrainer.evaluate(test).accuracy(), reports[2].validationAccuracy, 0.01);
}
