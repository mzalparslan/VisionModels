#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

#include "ImageDataset.h"

/**
 * @brief Generates 28x28 seven-segment digits (like a calculator display)
 * with random size, position, stroke width, brightness and noise.
 *
 * A stand-in for MNIST when it has not been downloaded, and a small, fast,
 * learnable task for the tests. Much easier than handwriting: a CNN gets
 * close to 100% on it.
 *
 *    aaa
 *   f   b
 *    ggg
 *   e   c
 *    ddd
 */
class SyntheticDigits {
public:
	/**
	 * @param count Number of images; labels cycle 0, 1, ..., 9.
	 * @param seed Same seed, same images.
	 */
	static ImageDataset generate(std::size_t count, std::uint32_t seed = 7) {
		// Segments a..g lit for each digit.
		static const bool segments[10][7] = {
			{ 1, 1, 1, 1, 1, 1, 0 }, // 0
			{ 0, 1, 1, 0, 0, 0, 0 }, // 1
			{ 1, 1, 0, 1, 1, 0, 1 }, // 2
			{ 1, 1, 1, 1, 0, 0, 1 }, // 3
			{ 0, 1, 1, 0, 0, 1, 1 }, // 4
			{ 1, 0, 1, 1, 0, 1, 1 }, // 5
			{ 1, 0, 1, 1, 1, 1, 1 }, // 6
			{ 1, 1, 1, 0, 0, 0, 0 }, // 7
			{ 1, 1, 1, 1, 1, 1, 1 }, // 8
			{ 1, 1, 1, 1, 0, 1, 1 }, // 9
		};

		std::mt19937 rng(seed);
		std::uniform_int_distribution<int> boxWidth(9, 14);
		std::uniform_int_distribution<int> boxHeight(15, 21);
		std::uniform_int_distribution<int> thickness(2, 3);
		std::uniform_real_distribution<float> ink(0.6f, 1.0f);
		std::normal_distribution<float> noise(0.0f, 0.08f);

		ImageDataset dataset;
		dataset.channels = 1;
		dataset.height = 28;
		dataset.width = 28;
		dataset.pixels.assign(count * 28 * 28, 0.0f);
		dataset.labels.resize(count);

		for (std::size_t i = 0; i < count; i++) {
			const int digit = static_cast<int>(i % 10);
			dataset.labels[i] = static_cast<std::uint8_t>(digit);
			float* image = &dataset.pixels[i * 28 * 28];

			const int w = boxWidth(rng);
			const int h = boxHeight(rng);
			const int t = thickness(rng);
			const int left = std::uniform_int_distribution<int>(2, 26 - w)(rng);
			const int top = std::uniform_int_distribution<int>(2, 26 - h)(rng);
			const int middle = top + h / 2;
			const float brightness = ink(rng);

			auto fill = [&](int x0, int y0, int x1, int y1) {
				for (int y = std::max(y0, 0); y < std::min(y1, 28); y++) {
					for (int x = std::max(x0, 0); x < std::min(x1, 28); x++) {
						image[y * 28 + x] = brightness;
					}
				}
			};

			const bool* lit = segments[digit];
			if (lit[0]) fill(left, top, left + w, top + t);                        // a
			if (lit[1]) fill(left + w - t, top, left + w, middle + 1);            // b
			if (lit[2]) fill(left + w - t, middle, left + w, top + h);            // c
			if (lit[3]) fill(left, top + h - t, left + w, top + h);               // d
			if (lit[4]) fill(left, middle, left + t, top + h);                    // e
			if (lit[5]) fill(left, top, left + t, middle + 1);                    // f
			if (lit[6]) fill(left, middle - t / 2, left + w, middle - t / 2 + t); // g

			for (std::size_t p = 0; p < 28 * 28; p++) {
				image[p] = std::clamp(image[p] + noise(rng), 0.0f, 1.0f);
			}
		}
		return dataset;
	}
};
