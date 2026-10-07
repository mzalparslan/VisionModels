#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "Validation.h"

/**
 * @brief Counts of (true class, predicted class) pairs, and the metrics
 * derived from them.
 *
 * Row = true class, column = predicted class; the diagonal is correct
 * predictions. For class k:
 *  - precision = correct k / everything predicted k   (how trustworthy a "k" is)
 *  - recall    = correct k / everything that really is k (how many k were found)
 *  - F1        = harmonic mean of the two.
 * For digits, the off-diagonal cells show which pairs get confused (4 vs 9,
 * 3 vs 5, 7 vs 1, ...).
 */
class ConfusionMatrix {
public:
	/**
	 * @throws InvalidParameterSizeError If classes is 0.
	 */
	explicit ConfusionMatrix(std::size_t classes)
		: classes(classes), counts(classes * classes, 0)
	{
		validation::requirePositiveSize(classes, "Class count");
	}

	/**
	 * @throws InvalidParameterError If a class is out of range.
	 */
	void add(std::size_t actual, std::size_t predicted) {
		validation::requireBelow(actual, classes, "Actual class");
		validation::requireBelow(predicted, classes, "Predicted class");
		counts.at(actual * classes + predicted)++;
	}

	std::size_t classCount() const { return classes; }
	std::size_t count(std::size_t actual, std::size_t predicted) const { return counts.at(actual * classes + predicted); }

	std::size_t total() const {
		std::size_t sum = 0;
		for (std::size_t c : counts) {
			sum += c;
		}
		return sum;
	}

	std::size_t correct() const {
		std::size_t sum = 0;
		for (std::size_t k = 0; k < classes; k++) {
			sum += count(k, k);
		}
		return sum;
	}

	/**
	 * @brief Fraction of all predictions that were right (0 when empty).
	 */
	double accuracy() const {
		const std::size_t all = total();
		return all == 0 ? 0.0 : static_cast<double>(correct()) / static_cast<double>(all);
	}

	double precision(std::size_t k) const {
		std::size_t predicted = 0;
		for (std::size_t a = 0; a < classes; a++) {
			predicted += count(a, k);
		}
		return predicted == 0 ? 0.0 : static_cast<double>(count(k, k)) / static_cast<double>(predicted);
	}

	double recall(std::size_t k) const {
		std::size_t actual = 0;
		for (std::size_t p = 0; p < classes; p++) {
			actual += count(k, p);
		}
		return actual == 0 ? 0.0 : static_cast<double>(count(k, k)) / static_cast<double>(actual);
	}

	double f1(std::size_t k) const {
		const double p = precision(k);
		const double r = recall(k);
		return (p + r) == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
	}

	/**
	 * @brief The matrix as a table, with recall per row and precision per column.
	 */
	std::string toString() const {
		std::ostringstream text;
		text << "actual\\pred";
		for (std::size_t p = 0; p < classes; p++) {
			text << std::setw(6) << p;
		}
		text << "  recall\n";
		for (std::size_t a = 0; a < classes; a++) {
			text << std::setw(11) << a;
			for (std::size_t p = 0; p < classes; p++) {
				text << std::setw(6) << count(a, p);
			}
			text << "  " << std::fixed << std::setprecision(3) << recall(a) << "\n";
		}
		text << "  precision";
		for (std::size_t p = 0; p < classes; p++) {
			text << " " << std::fixed << std::setprecision(3) << precision(p);
		}
		text << "\n";
		return text.str();
	}

private:
	std::size_t classes;
	std::vector<std::size_t> counts;
};
