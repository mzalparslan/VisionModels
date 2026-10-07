#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <ostream>
#include <random>
#include <vector>

#include "BenchmarkTimer.h"
#include "ConfusionMatrix.h"
#include "ConvNet.h"
#include "CudaRuntime.h"
#include "ExecutionStrategy.h"
#include "GpuConvNet.h"
#include "ImageDataset.h"
#include "Optimizer.h"
#include "ThreadPool.h"

/**
 * @brief Settings of a training run.
 */
class TrainingOptions {
public:
	// Full passes over the training set.
	std::size_t epochs = 2;
	// Images per optimizer step.
	std::size_t batchSize = 64;
	OptimizerSettings optimizer = OptimizerSettings::adam(1e-3);
	ExecutionStrategy strategy = ExecutionStrategy::Sequential;
	// Parallel only: threads to use (0 = the hardware thread count).
	std::size_t threads = 0;
	// Seed of the per-epoch shuffle. The same seed gives the same batches
	// whatever the strategy, so strategies can be compared.
	std::uint32_t shuffleSeed = 1;
	// Print the mean loss every this many steps (0 = only at the end of each epoch).
	std::size_t logEverySteps = 100;
};

/**
 * @brief What one epoch of training did.
 */
class EpochReport {
public:
	std::size_t epoch = 0;
	// Mean training loss over the epoch's batches.
	double trainingLoss = 0.0;
	std::uint64_t milliseconds = 0;
	// Accuracy on the validation set after the epoch (-1 when there is none).
	double validationAccuracy = -1.0;
};

/**
 * @brief Result of evaluate().
 */
class EvaluationReport {
public:
	ConfusionMatrix confusion;
	// Mean cross-entropy loss over the images.
	double loss = 0.0;
	// Predicted class of every image, in order.
	std::vector<std::size_t> predictions;

	explicit EvaluationReport(std::size_t classes) : confusion(classes) {}
	double accuracy() const { return confusion.accuracy(); }
};

/**
 * @brief Trains a ConvNet classifier on an ImageDataset and evaluates it,
 * with any ExecutionStrategy:
 *
 * @code
 * ConvNet<float> net = makeDigitCnn<float>();
 * TrainingOptions options;
 * options.strategy = ExecutionStrategy::Cuda;
 * ImageClassifierTrainer<float> trainer(net, options, std::cout);
 * trainer.train(trainingSet, &testSet);
 * EvaluationReport report = trainer.evaluate(testSet);
 * @endcode
 *
 *  - Sequential and Parallel train the ConvNet itself, with 1 or several threads.
 *  - Cuda copies the weights to a cuda::GpuConvNet, trains there with the
 *    training set held in GPU memory, and copies the weights back into the
 *    ConvNet after every epoch, so the model is always usable on the CPU too.
 *    Optimizer state (momentum, Adam's moments) stays on the device it was
 *    built on; switching strategy between calls starts it over.
 */
template <typename T>
class ImageClassifierTrainer {
public:
	/**
	 * @param model Trained in place; must outlive the trainer.
	 * @param log Where progress goes.
	 * @throws InvalidParameterError, InvalidParameterSizeError If options are invalid.
	 */
	ImageClassifierTrainer(ConvNet<T>& model, const TrainingOptions& options, std::ostream& log)
		: model(model), options(options), log(log)
	{
		validation::requirePositiveSize(options.epochs, "Epochs");
		validation::requirePositiveSize(options.batchSize, "Batch size");
		options.optimizer.validate();
	}

	/**
	 * @brief Trains for options.epochs epochs.
	 *
	 * @param validation Optional set evaluated after every epoch.
	 * @throws InvalidSizeError If the images do not fit the model.
	 * @throws CudaError With ExecutionStrategy::Cuda when no GPU is available.
	 */
	std::vector<EpochReport> train(const ImageDataset& training, const ImageDataset* validation = nullptr) {
		requireCompatible(training);
		if (validation != nullptr) {
			requireCompatible(*validation);
		}

		log << "Training " << model.parameterCount() << " parameters on " << training.size() << " images: "
			<< options.epochs << " epoch(s), batch " << options.batchSize << ", " << toString(options.optimizer.kind)
			<< " lr " << options.optimizer.learningRate << ", strategy " << toString(options.strategy);
		if (options.strategy == ExecutionStrategy::Parallel) {
			log << " (" << threadCount() << " threads)";
		}
		if (options.strategy == ExecutionStrategy::Cuda) {
			log << " (" << cuda::describeDevice() << ")";
		}
		log << std::endl;

		if (options.strategy == ExecutionStrategy::Cuda) {
			prepareGpu();
			gpu->uploadTrainingSet(training.pixels, training.labels);
		}

		std::mt19937 shuffleEngine(options.shuffleSeed);
		std::vector<EpochReport> reports;
		for (std::size_t epoch = 1; epoch <= options.epochs; epoch++) {
			BenchmarkTimer timer;
			timer.start();
			EpochReport report;
			report.epoch = epoch;
			report.trainingLoss = trainEpoch(training, shuffleEngine, epoch);
			if (options.strategy == ExecutionStrategy::Cuda) {
				downloadFromGpu();
			}
			report.milliseconds = timer.stop();

			log << "Epoch " << epoch << ": loss " << std::fixed << std::setprecision(4) << report.trainingLoss
				<< ", " << report.milliseconds << " ms";
			if (validation != nullptr) {
				report.validationAccuracy = evaluate(*validation).accuracy();
				log << ", validation accuracy " << std::setprecision(2) << 100.0 * report.validationAccuracy << "%";
			}
			log << std::defaultfloat << std::endl;
			reports.push_back(report);
		}
		return reports;
	}

	/**
	 * @brief Loss, accuracy, confusion matrix and predictions on a dataset,
	 * computed with the trainer's strategy.
	 *
	 * @throws InvalidSizeError If the images do not fit the model.
	 */
	EvaluationReport evaluate(const ImageDataset& data) {
		requireCompatible(data);
		const std::size_t classes = model.classCount();
		EvaluationReport report(classes);
		report.predictions.reserve(data.size());
		double lossSum = 0.0;

		if (options.strategy == ExecutionStrategy::Cuda) {
			prepareGpu();
			std::vector<float> logits;
			gpu->logits(data.pixels, data.size(), logits);
			Tensor<T> scores({ data.size(), classes });
			std::transform(logits.begin(), logits.end(), scores.data.begin(), [](float x) { return static_cast<T>(x); });
			accumulate(scores, data, 0, report, lossSum);
		}
		else {
			const std::size_t chunk = std::max<std::size_t>(options.batchSize, 256);
			for (std::size_t start = 0; start < data.size(); start += chunk) {
				const std::size_t count = std::min(chunk, data.size() - start);
				std::vector<std::size_t> indices(count);
				for (std::size_t i = 0; i < count; i++) {
					indices[i] = start + i;
				}
				Tensor<T> scores = model.forward(data.batchImages<T>(indices), threadCount());
				accumulate(scores, data, start, report, lossSum);
			}
		}
		report.loss = lossSum / static_cast<double>(data.size());
		return report;
	}

private:
	ConvNet<T>& model;
	TrainingOptions options;
	std::ostream& log;
	std::unique_ptr<cuda::GpuConvNet> gpu;

	std::size_t threadCount() const {
		if (options.strategy != ExecutionStrategy::Parallel) {
			return 1;
		}
		return options.threads == 0 ? ThreadPool::defaultThreadCount() : options.threads;
	}

	void requireCompatible(const ImageDataset& data) const {
		data.validate();
		validation::requirePositiveSize(data.size(), "Dataset size");
		if (data.sampleShape() != model.inputShape()) {
			throw InvalidSizeError("The dataset's images do not have the model's input shape!");
		}
		if (data.classCount() > model.classCount()) {
			throw InvalidSizeError("The dataset has more classes than the model outputs!");
		}
	}

	/**
	 * @return Mean loss of the epoch's batches.
	 */
	double trainEpoch(const ImageDataset& training, std::mt19937& shuffleEngine, std::size_t epoch) {
		const std::vector<std::size_t> order = training.shuffledIndices(shuffleEngine);
		const std::size_t batches = (order.size() + options.batchSize - 1) / options.batchSize;
		double epochLoss = 0.0;
		double windowLoss = 0.0;
		std::size_t windowSteps = 0;

		for (std::size_t b = 0; b < batches; b++) {
			const std::size_t start = b * options.batchSize;
			const std::size_t end = std::min(order.size(), start + options.batchSize);
			const std::vector<std::size_t> indices(order.begin() + start, order.begin() + end);

			double loss = 0.0;
			if (options.strategy == ExecutionStrategy::Cuda) {
				std::vector<std::uint32_t> gpuIndices(indices.size());
				std::transform(indices.begin(), indices.end(), gpuIndices.begin(),
					[](std::size_t index) { return static_cast<std::uint32_t>(index); });
				loss = gpu->computeGradientsResident(gpuIndices);
				gpu->update(options.optimizer);
			}
			else {
				loss = static_cast<double>(model.trainBatch(training.batchImages<T>(indices),
					training.batchLabels(indices), options.optimizer, threadCount()));
			}

			epochLoss += loss;
			windowLoss += loss;
			windowSteps++;
			if (options.logEverySteps > 0 && windowSteps == options.logEverySteps) {
				log << "  epoch " << epoch << ", step " << (b + 1) << "/" << batches << ": loss "
					<< std::fixed << std::setprecision(4) << windowLoss / static_cast<double>(windowSteps)
					<< std::defaultfloat << std::endl;
				windowLoss = 0.0;
				windowSteps = 0;
			}
		}
		return epochLoss / static_cast<double>(batches);
	}

	/**
	 * @brief Makes sure a GpuConvNet exists and holds the model's current weights.
	 */
	void prepareGpu() {
		if (!gpu) {
			if (!cuda::isAvailable()) {
				throw CudaError("ExecutionStrategy::Cuda needs a CUDA device: " + cuda::describeDevice() + ".");
			}
			gpu = std::make_unique<cuda::GpuConvNet>(model.specs(), model.inputShape(), options.batchSize);
			gpu->uploadParameters(model.template parameterValues<float>(), model.steps());
		}
	}

	void downloadFromGpu() {
		std::vector<std::vector<float>> values;
		gpu->downloadParameters(values);
		model.setParameterValues(values, gpu->steps());
	}

	void accumulate(const Tensor<T>& scores, const ImageDataset& data, std::size_t start,
		EvaluationReport& report, double& lossSum) {
		const std::size_t count = scores.shape[0];
		const std::vector<std::size_t> predicted = ConvNet<T>::argmaxRows(scores);
		std::vector<std::size_t> labels(count);
		for (std::size_t i = 0; i < count; i++) {
			labels[i] = data.labels[start + i];
			report.confusion.add(labels[i], predicted[i]);
			report.predictions.push_back(predicted[i]);
		}
		lossSum += static_cast<double>(SoftmaxCrossEntropy<T>::loss(scores, labels)) * static_cast<double>(count);
	}
};
