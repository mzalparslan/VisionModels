#pragma once

// The CUDA kernels of GpuConvNet. Each one gives one GPU thread one output
// number and mirrors a loop of the CPU layers (Conv2D.h, MaxPool2D.h, ...), so
// the two can be read side by side. Tensors are NCHW, as on the CPU.

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda {

	namespace kernels {

		/**
		 * @brief This thread's element index in a 1-D launch.
		 */
		__device__ inline std::size_t globalIndex() {
			return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		}

		// ------------------------------------------------------------------ Conv2D

		/**
		 * @brief One thread per output element (n, oc, oh, ow):
		 * out = bias[oc] + sum over ic, kh, kw of W[oc, ic, kh, kw] * in[n, ic, ih, iw].
		 */
		__global__ void conv2dForward(std::size_t total, const float* input, const float* weights, const float* bias,
			float* output, int inChannels, int height, int width, int outChannels, int outHeight, int outWidth,
			int kernelSize, int stride, int padding) {
			const std::size_t index = globalIndex();
			if (index >= total) {
				return;
			}
			const int ow = static_cast<int>(index % outWidth);
			const int oh = static_cast<int>((index / outWidth) % outHeight);
			const int oc = static_cast<int>((index / (static_cast<std::size_t>(outWidth) * outHeight)) % outChannels);
			const std::size_t n = index / (static_cast<std::size_t>(outWidth) * outHeight * outChannels);

			float sum = bias[oc];
			for (int ic = 0; ic < inChannels; ic++) {
				const float* image = input + (n * inChannels + ic) * height * width;
				const float* kernel = weights + (static_cast<std::size_t>(oc) * inChannels + ic) * kernelSize * kernelSize;
				for (int kh = 0; kh < kernelSize; kh++) {
					const int ih = oh * stride + kh - padding;
					if (ih < 0 || ih >= height) {
						continue;
					}
					for (int kw = 0; kw < kernelSize; kw++) {
						const int iw = ow * stride + kw - padding;
						if (iw < 0 || iw >= width) {
							continue;
						}
						sum += kernel[kh * kernelSize + kw] * image[ih * width + iw];
					}
				}
			}
			output[index] = sum;
		}

		/**
		 * @brief Input gradient, one thread per input element (n, ic, ih, iw).
		 *
		 * The CPU scatters each output gradient over its window. On the GPU that
		 * would need atomic adds, so each input pixel gathers instead: it visits
		 * every (oc, kh, kw) and works out which output position (oh, ow), if any,
		 * put kernel offset (kh, kw) on it: oh = (ih + padding - kh) / stride, when
		 * that divides evenly and is in range.
		 */
		__global__ void conv2dBackwardInput(std::size_t total, const float* gradOutput, const float* weights,
			float* gradInput, int inChannels, int height, int width, int outChannels, int outHeight, int outWidth,
			int kernelSize, int stride, int padding) {
			const std::size_t index = globalIndex();
			if (index >= total) {
				return;
			}
			const int iw = static_cast<int>(index % width);
			const int ih = static_cast<int>((index / width) % height);
			const int ic = static_cast<int>((index / (static_cast<std::size_t>(width) * height)) % inChannels);
			const std::size_t n = index / (static_cast<std::size_t>(width) * height * inChannels);

			float sum = 0.0f;
			for (int oc = 0; oc < outChannels; oc++) {
				const float* dOut = gradOutput + (n * outChannels + oc) * outHeight * outWidth;
				const float* kernel = weights + (static_cast<std::size_t>(oc) * inChannels + ic) * kernelSize * kernelSize;
				for (int kh = 0; kh < kernelSize; kh++) {
					const int rowOffset = ih + padding - kh;
					if (rowOffset < 0 || rowOffset % stride != 0) {
						continue;
					}
					const int oh = rowOffset / stride;
					if (oh >= outHeight) {
						continue;
					}
					for (int kw = 0; kw < kernelSize; kw++) {
						const int columnOffset = iw + padding - kw;
						if (columnOffset < 0 || columnOffset % stride != 0) {
							continue;
						}
						const int ow = columnOffset / stride;
						if (ow >= outWidth) {
							continue;
						}
						sum += dOut[oh * outWidth + ow] * kernel[kh * kernelSize + kw];
					}
				}
			}
			gradInput[index] = sum;
		}

		/**
		 * @brief Weight and bias gradients, first stage: one thread per
		 * (sample n, parameter j), summing over output positions of sample n only.
		 *
		 * Parameters j < weightCount are weights (oc, ic, kh, kw); the rest are
		 * biases. Summing over the whole batch in one thread would leave most of
		 * the GPU idle (conv1 has only 80 parameters), so the batch is split out
		 * here and summed by conv2dReduceBatch().
		 */
		__global__ void conv2dBackwardParametersPartial(std::size_t total, const float* input, const float* gradOutput,
			float* partial, int inChannels, int height, int width, int outChannels, int outHeight, int outWidth,
			int kernelSize, int stride, int padding) {
			const std::size_t index = globalIndex();
			if (index >= total) {
				return;
			}
			const std::size_t weightCount = static_cast<std::size_t>(outChannels) * inChannels * kernelSize * kernelSize;
			const std::size_t perSample = weightCount + outChannels;
			const std::size_t n = index / perSample;
			const std::size_t j = index % perSample;

			float sum = 0.0f;
			if (j < weightCount) {
				const int kw = static_cast<int>(j % kernelSize);
				const int kh = static_cast<int>((j / kernelSize) % kernelSize);
				const int ic = static_cast<int>((j / (static_cast<std::size_t>(kernelSize) * kernelSize)) % inChannels);
				const int oc = static_cast<int>(j / (static_cast<std::size_t>(kernelSize) * kernelSize * inChannels));
				const float* dOut = gradOutput + (n * outChannels + oc) * outHeight * outWidth;
				const float* image = input + (n * inChannels + ic) * height * width;
				for (int oh = 0; oh < outHeight; oh++) {
					const int ih = oh * stride + kh - padding;
					if (ih < 0 || ih >= height) {
						continue;
					}
					for (int ow = 0; ow < outWidth; ow++) {
						const int iw = ow * stride + kw - padding;
						if (iw < 0 || iw >= width) {
							continue;
						}
						sum += dOut[oh * outWidth + ow] * image[ih * width + iw];
					}
				}
			}
			else {
				const std::size_t oc = j - weightCount;
				const float* dOut = gradOutput + (n * outChannels + oc) * outHeight * outWidth;
				for (int p = 0; p < outHeight * outWidth; p++) {
					sum += dOut[p];
				}
			}
			partial[index] = sum;
		}

		/**
		 * @brief Second stage: one thread per parameter j sums the batch's
		 * partials, writing weights (j < weightCount) then biases.
		 */
		__global__ void reduceBatch(std::size_t perSample, std::size_t batch, const float* partial,
			std::size_t weightCount, float* weightGrad, float* biasGrad) {
			const std::size_t j = globalIndex();
			if (j >= perSample) {
				return;
			}
			float sum = 0.0f;
			for (std::size_t n = 0; n < batch; n++) {
				sum += partial[n * perSample + j];
			}
			if (j < weightCount) {
				weightGrad[j] = sum;
			}
			else {
				biasGrad[j - weightCount] = sum;
			}
		}

		// -------------------------------------------------------------------- ReLU

		__global__ void reluForward(std::size_t total, const float* input, float* output) {
			const std::size_t i = globalIndex();
			if (i < total) {
				output[i] = input[i] > 0.0f ? input[i] : 0.0f;
			}
		}

		__global__ void reluBackward(std::size_t total, const float* input, const float* gradOutput, float* gradInput) {
			const std::size_t i = globalIndex();
			if (i < total) {
				gradInput[i] = input[i] > 0.0f ? gradOutput[i] : 0.0f;
			}
		}

		// --------------------------------------------------------------- MaxPool2D

		/**
		 * @brief One thread per output element: the largest value of its window,
		 * and where it was (the first, if several tie).
		 */
		__global__ void maxPoolForward(std::size_t total, const float* input, float* output, int* winners,
			int height, int width, int outHeight, int outWidth, int kernelSize, int stride) {
			const std::size_t index = globalIndex();
			if (index >= total) {
				return;
			}
			const int ow = static_cast<int>(index % outWidth);
			const int oh = static_cast<int>((index / outWidth) % outHeight);
			const std::size_t plane = index / (static_cast<std::size_t>(outWidth) * outHeight);
			const std::size_t planeStart = plane * height * width;

			std::size_t best = planeStart + static_cast<std::size_t>(oh * stride) * width + ow * stride;
			for (int kh = 0; kh < kernelSize; kh++) {
				for (int kw = 0; kw < kernelSize; kw++) {
					const std::size_t candidate = planeStart + static_cast<std::size_t>(oh * stride + kh) * width + ow * stride + kw;
					if (input[candidate] > input[best]) {
						best = candidate;
					}
				}
			}
			output[index] = input[best];
			winners[index] = static_cast<int>(best);
		}

		/**
		 * @brief One thread per output element sends its gradient to its window's
		 * winner. gradInput must be zeroed first. Windows can overlap when
		 * stride < kernel, so two threads may add to the same input: atomicAdd.
		 */
		__global__ void maxPoolBackward(std::size_t total, const float* gradOutput, const int* winners, float* gradInput) {
			const std::size_t index = globalIndex();
			if (index < total) {
				atomicAdd(&gradInput[winners[index]], gradOutput[index]);
			}
		}

		// ------------------------------------------------------------------- Dense

		/**
		 * @brief One thread per (n, o): out = bias[o] + sum over i of in[n, i] * W[o, i].
		 */
		__global__ void denseForward(std::size_t total, const float* input, const float* weights, const float* bias,
			float* output, int inFeatures, int outFeatures) {
			const std::size_t index = globalIndex();
			if (index >= total) {
				return;
			}
			const std::size_t n = index / outFeatures;
			const int o = static_cast<int>(index % outFeatures);
			const float* x = input + n * inFeatures;
			const float* row = weights + static_cast<std::size_t>(o) * inFeatures;
			float sum = bias[o];
			for (int i = 0; i < inFeatures; i++) {
				sum += x[i] * row[i];
			}
			output[index] = sum;
		}

		/**
		 * @brief One thread per parameter: dW[o, i] = sum over n of dOut[n, o] * in[n, i],
		 * then (j >= weight count) dB[o] = sum over n of dOut[n, o].
		 */
		__global__ void denseBackwardParameters(std::size_t total, const float* input, const float* gradOutput,
			float* weightGrad, float* biasGrad, std::size_t batch, int inFeatures, int outFeatures) {
			const std::size_t j = globalIndex();
			if (j >= total) {
				return;
			}
			const std::size_t weightCount = static_cast<std::size_t>(outFeatures) * inFeatures;
			float sum = 0.0f;
			if (j < weightCount) {
				const std::size_t o = j / inFeatures;
				const std::size_t i = j % inFeatures;
				for (std::size_t n = 0; n < batch; n++) {
					sum += gradOutput[n * outFeatures + o] * input[n * inFeatures + i];
				}
				weightGrad[j] = sum;
			}
			else {
				const std::size_t o = j - weightCount;
				for (std::size_t n = 0; n < batch; n++) {
					sum += gradOutput[n * outFeatures + o];
				}
				biasGrad[o] = sum;
			}
		}

		/**
		 * @brief One thread per (n, i): dIn[n, i] = sum over o of dOut[n, o] * W[o, i].
		 */
		__global__ void denseBackwardInput(std::size_t total, const float* gradOutput, const float* weights,
			float* gradInput, int inFeatures, int outFeatures) {
			const std::size_t index = globalIndex();
			if (index >= total) {
				return;
			}
			const std::size_t n = index / inFeatures;
			const int i = static_cast<int>(index % inFeatures);
			float sum = 0.0f;
			for (int o = 0; o < outFeatures; o++) {
				sum += gradOutput[n * outFeatures + o] * weights[static_cast<std::size_t>(o) * inFeatures + i];
			}
			gradInput[index] = sum;
		}

		// ---------------------------------------------------- Softmax cross-entropy

		/**
		 * @brief One thread per sample: its loss -log softmax(z)[label], and the
		 * gradient (softmax(z) - onehot(label)) / batch of its logits.
		 */
		__global__ void softmaxCrossEntropy(std::size_t batch, const float* logits, const int* labels,
			float* gradLogits, float* losses, int classes) {
			const std::size_t n = globalIndex();
			if (n >= batch) {
				return;
			}
			const float* z = logits + n * classes;
			float* dz = gradLogits + n * classes;
			float maxLogit = z[0];
			for (int k = 1; k < classes; k++) {
				maxLogit = fmaxf(maxLogit, z[k]);
			}
			float sumExp = 0.0f;
			for (int k = 0; k < classes; k++) {
				dz[k] = expf(z[k] - maxLogit);
				sumExp += dz[k];
			}
			const int label = labels[n];
			losses[n] = logf(sumExp) - (z[label] - maxLogit);
			const float scale = 1.0f / static_cast<float>(batch);
			for (int k = 0; k < classes; k++) {
				dz[k] = (dz[k] / sumExp - (k == label ? 1.0f : 0.0f)) * scale;
			}
		}

		// -------------------------------------------------------------- Batching

		/**
		 * @brief Copies the images and labels at `indices` of the resident
		 * dataset into the batch buffers. One thread per batch pixel.
		 */
		__global__ void gatherBatch(std::size_t total, std::size_t sampleSize, const unsigned* indices,
			const float* setPixels, const int* setLabels, float* images, int* labels) {
			const std::size_t index = globalIndex();
			if (index >= total) {
				return;
			}
			const std::size_t b = index / sampleSize;
			const std::size_t p = index % sampleSize;
			const std::size_t source = indices[b];
			images[index] = setPixels[source * sampleSize + p];
			if (p == 0) {
				labels[b] = setLabels[source];
			}
		}

		// ------------------------------------------------------------ Optimizers

		__global__ void sgdUpdate(std::size_t count, float* value, const float* grad, float lr) {
			const std::size_t i = globalIndex();
			if (i < count) {
				value[i] -= lr * grad[i];
			}
		}

		__global__ void momentumUpdate(std::size_t count, float* value, const float* grad, float* velocity,
			float lr, float momentum) {
			const std::size_t i = globalIndex();
			if (i < count) {
				velocity[i] = momentum * velocity[i] + grad[i];
				value[i] -= lr * velocity[i];
			}
		}

		__global__ void adamUpdate(std::size_t count, float* value, const float* grad, float* m, float* v,
			float lr, float beta1, float beta2, float epsilon, float biasCorrection1, float biasCorrection2) {
			const std::size_t i = globalIndex();
			if (i < count) {
				const float g = grad[i];
				m[i] = beta1 * m[i] + (1.0f - beta1) * g;
				v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
				const float mHat = m[i] / biasCorrection1;
				const float vHat = v[i] / biasCorrection2;
				value[i] -= lr * mHat / (sqrtf(vHat) + epsilon);
			}
		}

	} // namespace kernels

} // namespace cuda
