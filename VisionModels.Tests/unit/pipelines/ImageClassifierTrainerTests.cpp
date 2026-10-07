#include "pch.h"

#include <sstream>

#include "DigitCnn.h"
#include "ImageClassifierTrainer.h"
#include "SyntheticDigits.h"

namespace {

	TrainingOptions quickOptions(ExecutionStrategy strategy) {
		TrainingOptions options;
		options.epochs = 1;
		options.batchSize = 32;
		options.optimizer = OptimizerSettings::adam(2e-3);
		options.strategy = strategy;
		options.threads = 8;
		options.logEverySteps = 0;
		return options;
	}

}

TEST(ImageClassifierTrainerTests, LearnsSyntheticDigits) {
	const ImageDataset training = SyntheticDigits::generate(3000, 1);
	const ImageDataset test = SyntheticDigits::generate(500, 2);
	ConvNet<float> net = makeDigitCnn<float>();
	std::ostringstream log;
	TrainingOptions options = quickOptions(ExecutionStrategy::Parallel);
	options.epochs = 3;
	options.optimizer = OptimizerSettings::adam(3e-3);
	ImageClassifierTrainer<float> trainer(net, options, log);

	const std::vector<EpochReport> reports = trainer.train(training, &test);
	ASSERT_EQ(reports.size(), 3u);
	EXPECT_LT(reports[2].trainingLoss, reports[0].trainingLoss);
	EXPECT_GT(reports[2].validationAccuracy, 0.9);

	const EvaluationReport report = trainer.evaluate(test);
	EXPECT_EQ(report.confusion.total(), test.size());
	EXPECT_EQ(report.predictions.size(), test.size());
	EXPECT_DOUBLE_EQ(report.accuracy(), reports[2].validationAccuracy);
	EXPECT_NE(log.str().find("Epoch 1"), std::string::npos);
}

TEST(ImageClassifierTrainerTests, SequentialAndParallelGiveTheSameModel) {
	const ImageDataset training = SyntheticDigits::generate(256, 3);
	ConvNet<float> sequentialNet = makeDigitCnn<float>();
	ConvNet<float> parallelNet = makeDigitCnn<float>();
	std::ostringstream log;

	ImageClassifierTrainer<float>(sequentialNet, quickOptions(ExecutionStrategy::Sequential), log).train(training);
	ImageClassifierTrainer<float>(parallelNet, quickOptions(ExecutionStrategy::Parallel), log).train(training);

	EXPECT_EQ(sequentialNet.parameterValues<float>(), parallelNet.parameterValues<float>());
}

TEST(ImageClassifierTrainerTests, RejectsMismatchedData) {
	ConvNet<float> net = makeDigitCnn<float>();
	std::ostringstream log;
	ImageClassifierTrainer<float> trainer(net, quickOptions(ExecutionStrategy::Sequential), log);
	ImageDataset wrong = SyntheticDigits::generate(10, 1);
	wrong.height = 14;
	wrong.width = 56;
	EXPECT_THROW(trainer.train(wrong), InvalidSizeError);

	TrainingOptions badOptions = quickOptions(ExecutionStrategy::Sequential);
	badOptions.batchSize = 0;
	EXPECT_THROW(ImageClassifierTrainer<float>(net, badOptions, log), InvalidParameterSizeError);
}

TEST(ImageClassifierTrainerTests, CudaWithoutGpuReportsCudaError) {
	if (cuda::isAvailable()) {
		return; // only meaningful where there is no GPU
	}
	ConvNet<float> net = makeDigitCnn<float>();
	std::ostringstream log;
	ImageClassifierTrainer<float> trainer(net, quickOptions(ExecutionStrategy::Cuda), log);
	EXPECT_THROW(trainer.train(SyntheticDigits::generate(32, 1)), CudaError);
}
