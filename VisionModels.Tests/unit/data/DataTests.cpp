#include "pch.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "ConfusionMatrix.h"
#include "MnistLoader.h"
#include "SyntheticDigits.h"

namespace {

	namespace fs = std::filesystem;

	void writeBigEndian(std::ofstream& file, std::uint32_t value) {
		const unsigned char bytes[4] = {
			static_cast<unsigned char>(value >> 24), static_cast<unsigned char>(value >> 16),
			static_cast<unsigned char>(value >> 8), static_cast<unsigned char>(value) };
		file.write(reinterpret_cast<const char*>(bytes), 4);
	}

	/**
	 * @brief A temporary folder with tiny IDX files: `count` 2x3 images whose
	 * pixels are image * 10 + pixel, and labels image % 10.
	 */
	class IdxFolder {
	public:
		fs::path directory;

		explicit IdxFolder(std::uint32_t count, std::uint32_t labelCount, std::uint32_t imageMagic = 0x00000803) {
			directory = fs::temp_directory_path() / ("VisionModelsIdx" + std::to_string(std::random_device{}()));
			fs::create_directories(directory);

			std::ofstream images(directory / "train-images-idx3-ubyte", std::ios::binary);
			writeBigEndian(images, imageMagic);
			writeBigEndian(images, count);
			writeBigEndian(images, 2);
			writeBigEndian(images, 3);
			for (std::uint32_t i = 0; i < count; i++) {
				for (std::uint32_t p = 0; p < 6; p++) {
					const char value = static_cast<char>(i * 10 + p);
					images.write(&value, 1);
				}
			}

			std::ofstream labels(directory / "train-labels.idx1-ubyte", std::ios::binary);
			writeBigEndian(labels, 0x00000801);
			writeBigEndian(labels, labelCount);
			for (std::uint32_t i = 0; i < labelCount; i++) {
				const char value = static_cast<char>(i % 10);
				labels.write(&value, 1);
			}
		}

		~IdxFolder() {
			std::error_code ignored;
			fs::remove_all(directory, ignored);
		}
	};

}

// -------------------------------------------------------------- MnistLoader

TEST(MnistLoaderTests, ReadsIdxFilesUnderEitherName) {
	ImageDataset data;
	{
		IdxFolder folder(3, 3);
		data = MnistLoader::loadTraining(folder.directory);
	}
	ASSERT_EQ(data.size(), 3u);
	EXPECT_EQ(data.height, 2u);
	EXPECT_EQ(data.width, 3u);
	EXPECT_EQ(data.labels, (std::vector<std::uint8_t>{ 0, 1, 2 }));
	EXPECT_FLOAT_EQ(data.pixels[6 + 4], 14.0f / 255.0f);
}

TEST(MnistLoaderTests, LimitLoadsTheFirstImagesOnly) {
	IdxFolder folder(5, 5);
	const ImageDataset data = MnistLoader::loadTraining(folder.directory, 2);
	EXPECT_EQ(data.size(), 2u);
	EXPECT_EQ(data.pixels.size(), 12u);
}

TEST(MnistLoaderTests, ReportsBadFiles) {
	{
		IdxFolder folder(3, 4);
		EXPECT_THROW(MnistLoader::loadTraining(folder.directory), DataLoadError);
	}
	{
		IdxFolder folder(3, 3, 0x00000801);
		EXPECT_THROW(MnistLoader::loadTraining(folder.directory), DataLoadError);
	}
	EXPECT_THROW(MnistLoader::loadTraining(fs::temp_directory_path() / "no-such-mnist"), DataLoadError);
	EXPECT_FALSE(MnistLoader::isAvailable(fs::temp_directory_path() / "no-such-mnist"));
}

// ------------------------------------------------------------- ImageDataset

TEST(ImageDatasetTests, BatchesAndHead) {
	const ImageDataset data = SyntheticDigits::generate(20, 1);
	const Tensor<double> batch = data.batchImages<double>({ 3, 0 });
	EXPECT_EQ(batch.shape, (std::vector<std::size_t>{ 2, 1, 28, 28 }));
	EXPECT_EQ(batch[0], static_cast<double>(data.pixels[3 * 784]));
	EXPECT_EQ(data.batchLabels({ 3, 0 }), (std::vector<std::size_t>{ 3, 0 }));
	EXPECT_THROW(data.batchLabels({ 20 }), InvalidParameterError);
	EXPECT_EQ(data.head(5).size(), 5u);
	EXPECT_EQ(data.classCount(), 10u);
	EXPECT_EQ(data.asciiArt(0).size(), 28u * 29u);
}

TEST(SyntheticDigitsTests, IsReproducibleAndInRange) {
	const ImageDataset first = SyntheticDigits::generate(30, 4);
	const ImageDataset second = SyntheticDigits::generate(30, 4);
	EXPECT_EQ(first.pixels, second.pixels);
	for (float pixel : first.pixels) {
		EXPECT_GE(pixel, 0.0f);
		EXPECT_LE(pixel, 1.0f);
	}
	EXPECT_NE(first.pixels, SyntheticDigits::generate(30, 5).pixels);
}

// ---------------------------------------------------------- ConfusionMatrix

TEST(ConfusionMatrixTests, AccuracyPrecisionRecall) {
	ConfusionMatrix matrix(3);
	matrix.add(0, 0);
	matrix.add(0, 0);
	matrix.add(0, 1);
	matrix.add(1, 1);
	matrix.add(2, 1);
	EXPECT_EQ(matrix.total(), 5u);
	EXPECT_DOUBLE_EQ(matrix.accuracy(), 3.0 / 5.0);
	EXPECT_DOUBLE_EQ(matrix.recall(0), 2.0 / 3.0);
	EXPECT_DOUBLE_EQ(matrix.precision(1), 1.0 / 3.0);
	EXPECT_DOUBLE_EQ(matrix.precision(2), 0.0);
	EXPECT_DOUBLE_EQ(matrix.f1(0), 0.8);
	EXPECT_THROW(matrix.add(3, 0), InvalidParameterError);
}
