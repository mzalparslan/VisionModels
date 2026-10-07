#pragma once

#include <cstddef>

#include "ThreadPool.h"

/**
 * @brief Runs body(begin, end) over [0, count): on the calling thread when
 * threads is 1, otherwise split across ThreadPool::shared().
 *
 * Every layer writes its loops as "process items [begin, end)", where each
 * item owns its outputs (an output feature map, a sample, a filter's
 * gradient) and sums its inputs in a fixed order. The same function then
 * serves Sequential (one range) and Parallel (several ranges) and both give
 * bit-identical results.
 *
 * @param count Number of independent items.
 * @param minPerChunk Fewest items worth giving to one thread.
 * @param threads Most threads to use (1 = Sequential).
 */
template <typename Body>
void parallelFor(std::size_t count, std::size_t minPerChunk, std::size_t threads, const Body& body) {
	if (count == 0) {
		return;
	}
	if (threads <= 1) {
		body(std::size_t(0), count);
		return;
	}
	ThreadPool::shared().parallelFor(count, minPerChunk, threads,
		[&body](std::size_t begin, std::size_t end) { body(begin, end); });
}
