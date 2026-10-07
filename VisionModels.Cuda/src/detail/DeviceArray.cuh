#pragma once

#include "CudaHost.cuh"

#include <cstddef>
#include <vector>

namespace cuda {

	namespace detail {

		/**
		 * @brief An array in GPU memory that is freed when the object goes away,
		 * so an exception between allocation and use cannot leak it.
		 */
		template <typename E>
		class DeviceArray {
		public:
			DeviceArray() = default;
			explicit DeviceArray(std::size_t count) { allocate(count); }
			~DeviceArray() { release(); }

			DeviceArray(const DeviceArray&) = delete;
			DeviceArray& operator=(const DeviceArray&) = delete;
			DeviceArray(DeviceArray&& other) noexcept : pointer(other.pointer), length(other.length) {
				other.pointer = nullptr;
				other.length = 0;
			}
			DeviceArray& operator=(DeviceArray&& other) noexcept {
				if (this != &other) {
					release();
					pointer = other.pointer;
					length = other.length;
					other.pointer = nullptr;
					other.length = 0;
				}
				return *this;
			}

			/**
			 * @brief Replaces the contents with `count` uninitialized elements.
			 */
			void allocate(std::size_t count) {
				release();
				if (count > 0) {
					check(cudaMalloc(&pointer, count * sizeof(E)), "cudaMalloc");
				}
				length = count;
			}

			void zero() {
				if (length > 0) {
					check(cudaMemset(pointer, 0, length * sizeof(E)), "cudaMemset");
				}
			}

			/**
			 * @brief Copies `count` elements from host memory to the start of the array.
			 */
			void upload(const E* source, std::size_t count) {
				if (count > 0) {
					check(cudaMemcpy(pointer, source, count * sizeof(E), cudaMemcpyHostToDevice), "cudaMemcpy (to device)");
				}
			}

			/**
			 * @brief Copies the first `count` elements to host memory. Waits for
			 * earlier kernels to finish first.
			 */
			void download(E* destination, std::size_t count) const {
				if (count > 0) {
					check(cudaMemcpy(destination, pointer, count * sizeof(E), cudaMemcpyDeviceToHost), "cudaMemcpy (to host)");
				}
			}

			std::vector<E> toVector() const {
				std::vector<E> result(length);
				download(result.data(), length);
				return result;
			}

			E* get() const { return pointer; }
			std::size_t size() const { return length; }

		private:
			E* pointer = nullptr;
			std::size_t length = 0;

			void release() {
				if (pointer != nullptr) {
					cudaFree(pointer);
					pointer = nullptr;
				}
				length = 0;
			}
		};

	} // namespace detail

} // namespace cuda
