# The CUDA version

`cuda::GpuConvNet` (`VisionModels/include/cuda/GpuConvNet.h`, implemented in
`VisionModels.Cuda/src/GpuConvNet.cu`) trains the same network as the CPU
`ConvNet`, entirely on the GPU.

## Keeping the data on the GPU

Copies between CPU and GPU memory are slow compared with the arithmetic on
either side, so the design keeps them to a minimum:

| Where | What | When it crosses |
|---|---|---|
| GPU | weights, gradients, momentum / Adam state | up once at the start, down after each epoch |
| GPU | activations and their gradients, one buffer per layer sized for the largest batch | never |
| GPU | the whole training set (60,000 x 784 floats = 188 MB) | up once |
| CPU -> GPU | the batch's 64 image indices | every step (256 bytes) |
| GPU -> CPU | the batch's 64 losses | every step (256 bytes) |

A `gatherBatch` kernel copies the chosen images out of the resident training
set into the input buffer. The shuffling stays on the CPU, using the same
random engine as the CPU strategies, so all three strategies see the same
batches.

## One thread per output number

Every kernel (`src/detail/ConvNetKernels.cuh`) gives one GPU thread one number
to produce and mirrors a loop of the CPU layer:

| Kernel | One thread computes |
|---|---|
| `conv2dForward` | one output pixel of one feature map |
| `conv2dBackwardInput` | one input pixel's gradient |
| `conv2dBackwardParametersPartial` + `reduceBatch` | one weight's gradient over one sample, then summed over the batch |
| `maxPoolForward` / `maxPoolBackward` | one pooling window |
| `denseForward`, `denseBackwardParameters`, `denseBackwardInput` | one output, one weight gradient, one input gradient |
| `softmaxCrossEntropy` | one sample's loss and logit gradients |
| `sgdUpdate`, `momentumUpdate`, `adamUpdate` | one weight |

Two places differ from the CPU loops, both to avoid many threads adding into
the same memory:

- **Convolution input gradient.** The CPU *scatters*: each output gradient is
  added to every input pixel under its window. On the GPU that would need atomic
  adds. Instead each thread takes one input pixel and *gathers* from the output
  positions whose window covered it: `oh = (ih + padding - kh) / stride`, when
  that divides evenly and is in range.
- **Convolution weight gradient.** One thread per weight summing over the whole
  batch would use only 80 threads for the first layer and leave the GPU nearly
  idle. The sum is split in two: one thread per (sample, weight), then one per
  weight summing across the batch.

Max pooling's backward pass does use `atomicAdd`, because overlapping windows
(stride < kernel) can share a winner; with 2x2 windows and stride 2 the
additions never collide.

## Accuracy

The GPU computes in float, sums in a different order than the CPU, and fuses
multiplies and adds (FMA). Its results therefore match the CPU `ConvNet<float>`
to within about 1e-4 relative, not bit for bit. The unit tests check this for
logits, gradients and optimizer steps. Training curves on MNIST are
indistinguishable (loss 0.2251 after one epoch with every strategy).

## Building

The `VisionModels.Cuda` project compiles `.cu` files through `nvcc-build.cmd`,
which sets up the MSVC v143 environment for nvcc (CUDA 13 does not accept
MSVC 19.50+ as its host compiler) and targets compute capability 89 by default:

```bash
msbuild VisionModels.sln /p:Configuration=Release /p:Platform=x64 /p:CudaComputeCapability=86
```

Without the CUDA Toolkit the project compiles `CudaUnavailable.cpp` and
`GpuConvNetUnavailable.cpp` instead, so everything links and `Cuda` reports
that it is not available.

## Where the time goes, and what would make it faster

One epoch takes about 0.47 s, roughly 0.5 ms per step of 64 images. At this
size the GPU is mostly waiting on kernel launches (about 35 per step: 9 forward,
17 backward, 1 gather, 8 optimizer) and the per-step loss copy, not on arithmetic. Things to try:

- launch with CUDA Graphs, or fuse Conv2D + ReLU and Dense + ReLU;
- larger batches (e.g. `--batch 256`);
- shared-memory tiling in the convolution and dense kernels, or im2col plus a
  tiled matrix multiply;
- profile it with Nsight Compute (already installed) to see which kernels dominate.
