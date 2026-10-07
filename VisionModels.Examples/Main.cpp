// VisionModels.Examples: trains the digit-recognition CNN (DigitCnn.h) on MNIST
// with a chosen ExecutionStrategy, then evaluates it on the test set.
//
// Without the MNIST files (scripts/download_mnist.ps1 or .sh puts them in
// resources/mnist), it falls back to generated seven-segment digits.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "CudaRuntime.h"
#include "DigitCnn.h"
#include "ExecutionStrategy.h"
#include "ImageClassifierTrainer.h"
#include "MnistLoader.h"
#include "SyntheticDigits.h"
#include "ThreadPool.h"

namespace {

	class Arguments {
	public:
		std::filesystem::path dataDirectory;
		bool synthetic = false;
		// "sequential", "parallel", "cuda", or "all" to compare them.
		std::string strategy = "parallel";
		std::size_t epochs = 2;
		std::size_t batchSize = 64;
		// Use only the first N training images (0 = all).
		std::size_t limit = 0;
		double learningRate = 1e-3;
		std::string optimizer = "adam";
		std::size_t threads = 0;
		// Misclassified test images to draw.
		std::size_t showMistakes = 3;
	};

	void printUsage() {
		std::cout
			<< "Usage: VisionModels.Examples [options]\n"
			<< "  --strategy S    sequential | parallel | cuda | all  (default parallel;\n"
			<< "                  'all' trains the same model with each and compares them)\n"
			<< "  --epochs N      passes over the training set (default 2)\n"
			<< "  --batch N       images per step (default 64)\n"
			<< "  --limit N       train on the first N images only (default: all)\n"
			<< "  --lr X          learning rate (default 0.001)\n"
			<< "  --optimizer O   adam | momentum | sgd (default adam)\n"
			<< "  --threads N     threads for parallel (default: hardware count)\n"
			<< "  --data DIR      folder with the MNIST files (default: resources/mnist)\n"
			<< "  --synthetic     use generated seven-segment digits instead of MNIST\n"
			<< "  --show N        draw N misclassified test images (default 3)\n";
	}

	std::size_t parseCount(const std::string& text, const std::string& option) {
		try {
			std::size_t used = 0;
			const unsigned long long value = std::stoull(text, &used);
			if (used == text.size()) {
				return static_cast<std::size_t>(value);
			}
		}
		catch (const std::exception&) {
		}
		throw InvalidParameterError(option + " needs a whole number, not '" + text + "'");
	}

	/**
	 * @brief resources/mnist in the working directory or one of its parents
	 * (the examples run from bin/, the solution folder or build/).
	 */
	std::filesystem::path findMnist() {
		std::filesystem::path directory = std::filesystem::current_path();
		for (int level = 0; level < 8; level++) {
			if (MnistLoader::isAvailable(directory / "resources" / "mnist")) {
				return directory / "resources" / "mnist";
			}
			if (!directory.has_parent_path() || directory.parent_path() == directory) {
				break;
			}
			directory = directory.parent_path();
		}
		return {};
	}

	OptimizerSettings makeOptimizer(const Arguments& arguments) {
		if (arguments.optimizer == "adam") {
			return OptimizerSettings::adam(arguments.learningRate);
		}
		if (arguments.optimizer == "momentum") {
			return OptimizerSettings::withMomentum(arguments.learningRate);
		}
		if (arguments.optimizer == "sgd") {
			return OptimizerSettings::sgd(arguments.learningRate);
		}
		throw InvalidParameterError("Unknown optimizer '" + arguments.optimizer + "'");
	}

	void showMistakes(const ImageDataset& test, const EvaluationReport& report, std::size_t count) {
		std::size_t shown = 0;
		for (std::size_t i = 0; i < test.size() && shown < count; i++) {
			if (report.predictions[i] != test.labels[i]) {
				std::cout << "\nTest image " << i << ": label " << int(test.labels[i])
					<< ", predicted " << report.predictions[i] << "\n" << test.asciiArt(i);
				shown++;
			}
		}
	}

	/**
	 * @brief Trains a fresh DigitCnn with one strategy and evaluates it.
	 */
	EvaluationReport trainAndEvaluate(const ImageDataset& training, const ImageDataset& test,
		const Arguments& arguments, ExecutionStrategy strategy, std::uint64_t& trainingMilliseconds) {
		ConvNet<float> net = makeDigitCnn<float>();

		TrainingOptions options;
		options.epochs = arguments.epochs;
		options.batchSize = arguments.batchSize;
		options.optimizer = makeOptimizer(arguments);
		options.strategy = strategy;
		options.threads = arguments.threads;
		options.logEverySteps = 200;

		ImageClassifierTrainer<float> trainer(net, options, std::cout);
		const std::vector<EpochReport> epochs = trainer.train(training);
		trainingMilliseconds = 0;
		for (const EpochReport& epoch : epochs) {
			trainingMilliseconds += epoch.milliseconds;
		}
		return trainer.evaluate(test);
	}

	int run(const Arguments& arguments) {
		ImageDataset training;
		ImageDataset test;
		const std::filesystem::path mnist = arguments.dataDirectory.empty() ? findMnist() : arguments.dataDirectory;
		if (!arguments.synthetic && !mnist.empty() && MnistLoader::isAvailable(mnist)) {
			std::cout << "MNIST from " << mnist.string() << std::endl;
			training = MnistLoader::loadTraining(mnist, arguments.limit);
			test = MnistLoader::loadTest(mnist);
		}
		else {
			if (!arguments.synthetic) {
				std::cout << "MNIST not found (run scripts/download_mnist.ps1 or scripts/download_mnist.sh);"
					<< " using generated seven-segment digits instead." << std::endl;
			}
			training = SyntheticDigits::generate(arguments.limit == 0 ? 10000 : arguments.limit, 7);
			test = SyntheticDigits::generate(2000, 8);
		}
		std::cout << training.size() << " training and " << test.size() << " test images of "
			<< training.height << "x" << training.width << "\n"
			<< "CPU: " << ThreadPool::defaultThreadCount() << " hardware threads; GPU: "
			<< cuda::describeDevice() << "\n\n";

		ConvNet<float> preview = makeDigitCnn<float>();
		std::cout << preview.summary() << std::endl;

		if (arguments.strategy == "all") {
			std::vector<ExecutionStrategy> strategies = { ExecutionStrategy::Sequential, ExecutionStrategy::Parallel };
			if (cuda::isAvailable()) {
				strategies.push_back(ExecutionStrategy::Cuda);
			}
			std::vector<std::uint64_t> times;
			std::vector<double> accuracies;
			for (ExecutionStrategy strategy : strategies) {
				std::cout << "\n===== " << toString(strategy) << " =====" << std::endl;
				std::uint64_t milliseconds = 0;
				accuracies.push_back(trainAndEvaluate(training, test, arguments, strategy, milliseconds).accuracy());
				times.push_back(milliseconds);
			}
			std::cout << "\nStrategy      training time   speedup   test accuracy\n";
			for (std::size_t s = 0; s < strategies.size(); s++) {
				std::cout << std::left << std::setw(12) << toString(strategies[s]) << std::right
					<< std::setw(12) << times[s] << " ms"
					<< std::setw(9) << std::fixed << std::setprecision(1)
					<< static_cast<double>(times[0]) / static_cast<double>(std::max<std::uint64_t>(times[s], 1)) << "x"
					<< std::setw(14) << std::setprecision(2) << 100.0 * accuracies[s] << "%\n";
			}
			return 0;
		}

		std::uint64_t milliseconds = 0;
		const EvaluationReport report = trainAndEvaluate(training, test, arguments,
			parseExecutionStrategy(arguments.strategy), milliseconds);
		std::cout << "\nTest accuracy " << std::fixed << std::setprecision(2) << 100.0 * report.accuracy()
			<< "% (" << report.confusion.correct() << "/" << report.confusion.total() << "), loss "
			<< std::setprecision(4) << report.loss << ", trained in " << milliseconds << " ms\n\n"
			<< report.confusion.toString();
		showMistakes(test, report, arguments.showMistakes);
		return 0;
	}

} // namespace

int main(int argc, char* argv[]) {
	Arguments arguments;
	try {
		for (int i = 1; i < argc; i++) {
			const std::string option = argv[i];
			auto value = [&]() -> std::string {
				if (i + 1 >= argc) {
					throw InvalidParameterError(option + " needs a value");
				}
				return argv[++i];
			};
			if (option == "--strategy") arguments.strategy = value();
			else if (option == "--epochs") arguments.epochs = parseCount(value(), option);
			else if (option == "--batch") arguments.batchSize = parseCount(value(), option);
			else if (option == "--limit") arguments.limit = parseCount(value(), option);
			else if (option == "--lr") arguments.learningRate = std::stod(value());
			else if (option == "--optimizer") arguments.optimizer = value();
			else if (option == "--threads") arguments.threads = parseCount(value(), option);
			else if (option == "--data") arguments.dataDirectory = value();
			else if (option == "--synthetic") arguments.synthetic = true;
			else if (option == "--show") arguments.showMistakes = parseCount(value(), option);
			else if (option == "--help" || option == "-h") {
				printUsage();
				return 0;
			}
			else {
				std::cout << "Unknown option " << option << "\n\n";
				printUsage();
				return 1;
			}
		}
		return run(arguments);
	}
	catch (const std::exception& error) {
		std::cerr << "Error: " << error.what() << std::endl;
		return 1;
	}
}
