#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "Exceptions.h"
#include "ImageDataset.h"

/**
 * @brief Reads the MNIST handwritten digit dataset (LeCun, Cortes and Burges):
 * 60,000 training and 10,000 test images of 28x28 gray pixels, labels 0-9.
 *
 * The files use the IDX format: a big-endian header, then raw bytes.
 *   images: magic 0x00000803, count, rows, columns, then count*rows*columns
 *           pixels (0 = background, 255 = ink), row by row;
 *   labels: magic 0x00000801, count, then count labels.
 * The files must be uncompressed (the downloads are .gz; scripts/download_mnist
 * unpacks them). Pixels are scaled to [0, 1].
 */
class MnistLoader {
public:
	/**
	 * @brief Loads one image file and its label file.
	 *
	 * @param limit Load only the first `limit` images (0 = all).
	 * @throws DataLoadError If a file is missing, truncated, or not IDX data
	 * of the right kind, or the two files disagree on the count.
	 */
	static ImageDataset load(const std::filesystem::path& imagesPath,
		const std::filesystem::path& labelsPath, std::size_t limit = 0) {
		std::ifstream images = open(imagesPath);
		std::ifstream labels = open(labelsPath);

		const std::uint32_t imageMagic = readBigEndian(images, imagesPath);
		if (imageMagic != 0x00000803) {
			throw DataLoadError(imagesPath.string() + " is not an IDX image file (magic "
				+ std::to_string(imageMagic) + ")!");
		}
		const std::uint32_t imageCount = readBigEndian(images, imagesPath);
		const std::uint32_t rows = readBigEndian(images, imagesPath);
		const std::uint32_t columns = readBigEndian(images, imagesPath);

		const std::uint32_t labelMagic = readBigEndian(labels, labelsPath);
		if (labelMagic != 0x00000801) {
			throw DataLoadError(labelsPath.string() + " is not an IDX label file (magic "
				+ std::to_string(labelMagic) + ")!");
		}
		const std::uint32_t labelCount = readBigEndian(labels, labelsPath);
		if (labelCount != imageCount) {
			throw DataLoadError("MNIST: " + std::to_string(imageCount) + " images but "
				+ std::to_string(labelCount) + " labels!");
		}
		if (imageCount == 0 || rows == 0 || columns == 0) {
			throw DataLoadError(imagesPath.string() + " holds no images!");
		}

		const std::size_t count = (limit == 0 || limit > imageCount) ? imageCount : limit;
		ImageDataset dataset;
		dataset.channels = 1;
		dataset.height = rows;
		dataset.width = columns;

		std::vector<unsigned char> raw(count * rows * columns);
		if (!images.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()))) {
			throw DataLoadError(imagesPath.string() + " is truncated!");
		}
		dataset.pixels.resize(raw.size());
		for (std::size_t i = 0; i < raw.size(); i++) {
			dataset.pixels[i] = static_cast<float>(raw[i]) / 255.0f;
		}

		dataset.labels.resize(count);
		if (!labels.read(reinterpret_cast<char*>(dataset.labels.data()), static_cast<std::streamsize>(count))) {
			throw DataLoadError(labelsPath.string() + " is truncated!");
		}
		for (std::uint8_t label : dataset.labels) {
			if (label > 9) {
				throw DataLoadError(labelsPath.string() + " holds a label above 9!");
			}
		}
		return dataset;
	}

	/**
	 * @brief The 60,000-image training set from `directory`.
	 * @throws DataLoadError See load().
	 */
	static ImageDataset loadTraining(const std::filesystem::path& directory, std::size_t limit = 0) {
		return load(find(directory, "train-images"), find(directory, "train-labels"), limit);
	}

	/**
	 * @brief The 10,000-image test set from `directory`.
	 * @throws DataLoadError See load().
	 */
	static ImageDataset loadTest(const std::filesystem::path& directory, std::size_t limit = 0) {
		return load(find(directory, "t10k-images"), find(directory, "t10k-labels"), limit);
	}

	/**
	 * @brief Whether all four MNIST files are in `directory`. Never throws.
	 */
	static bool isAvailable(const std::filesystem::path& directory) {
		try {
			find(directory, "train-images");
			find(directory, "train-labels");
			find(directory, "t10k-images");
			find(directory, "t10k-labels");
			return true;
		}
		catch (const DataLoadError&) {
			return false;
		}
	}

private:
	/**
	 * @brief The file for `stem`, under either spelling in use
	 * ("train-images-idx3-ubyte" or "train-images.idx3-ubyte").
	 */
	static std::filesystem::path find(const std::filesystem::path& directory, const std::string& stem) {
		const std::string kind = stem.find("images") != std::string::npos ? "idx3-ubyte" : "idx1-ubyte";
		for (const std::string& name : { stem + "-" + kind, stem + "." + kind }) {
			std::error_code error;
			if (std::filesystem::is_regular_file(directory / name, error)) {
				return directory / name;
			}
		}
		throw DataLoadError("MNIST file " + stem + "-" + kind + " not found in " + directory.string()
			+ " (run scripts/download_mnist.ps1 or scripts/download_mnist.sh)");
	}

	static std::ifstream open(const std::filesystem::path& path) {
		std::ifstream file(path, std::ios::binary);
		if (!file) {
			throw DataLoadError("Cannot open " + path.string() + "!");
		}
		return file;
	}

	static std::uint32_t readBigEndian(std::ifstream& file, const std::filesystem::path& path) {
		unsigned char bytes[4];
		if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
			throw DataLoadError(path.string() + " is too short for an IDX header!");
		}
		return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16)
			| (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
	}
};
