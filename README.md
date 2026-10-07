# VisionModels

A from-scratch **C++20** implementation of convolutional neural networks (CNNs), 
trained to recognize handwritten digits (MNIST): classic first computer-vision task.

There are no dependencies and no framework. Tensors, convolution, pooling,
backpropagation and optimizers are all written by hand, and every gradient is
checked against finite differences in test suite. Google Test is used for
tests only.

Same network trains in three ways (`ExecutionStrategy`):

| Strategy | How | One MNIST Epoch (60,000 images) | Speedup | Test accuracy after 1 Epoch |
|---|---|---|---|---|
| `Sequential` | one CPU thread | 94.4 s | 1x | 97.53% |
| `Parallel` | 16 CPU threads, Parallel Runs | 19.7 s | 4.8x | 97.53% |
| `Cuda` | NVIDIA GPU, everything resident in GPU memory | 0.48 s | 197x | 97.58% |

Release build, batch 64, Adam (lr 0.001), RTX 4060 Ti and a 16-thread CPU. Ten
epochs on GPU (4.7 s) reach **98.82%** test accuracy.

> This is a learning project. Kernels are direct "naive" algorithms,
> written as self study projects, not to compete with cuDNN or oneDNN.

## Contents

- [CUDA Implementation](#cuda-implementation)
- [Getting Started](#getting-started)
- [Repository Layout](#repository-layout)
- [Using Library](#using-library)
- [Model](#model)
- [How Three Strategies Work](#how-three-strategies-work)
- [Tests](#tests)
- [Future Ideas](#future-ideas)
- [License](#license)

## CUDA Implementation

`cuda::GpuConvNet` (`VisionModels/include/cuda/GpuConvNet.h`, implemented in
`VisionModels.Cuda/src/GpuConvNet.cu`) trains same network as CPU `ConvNet`, 
entirely on GPU.

### Keeping Data on GPU

Copies between CPU and GPU memory are slow compared with arithmetic on
either side, so design keeps them to a minimum:

| Where | What | When it crosses |
|---|---|---|
| GPU | weights, gradients, momentum / Adam state | up once at start, down after each epoch |
| GPU | activations and their gradients, one buffer per layer sized for largest batch | never |
| GPU | Whole training set (60,000 x 784 floats = 188 MB) | up once |
| CPU -> GPU | Batch's 64 image indices | every step (256 bytes) |
| GPU -> CPU | Batch's 64 losses | every step (256 bytes) |

A `gatherBatch` kernel copies chosen images out of resident training
set into input buffer. Shuffling stays on CPU, using same random 
engine as CPU strategies, so all three strategies see same batches.

### One Thread per Output Number

Every kernel (`src/detail/ConvNetKernels.cuh`) gives one GPU thread one number
to produce and mirrors a loop of CPU layer:

| Kernel | One thread computes |
|---|---|
| `conv2dForward` | one output pixel of one feature map |
| `conv2dBackwardInput` | one input pixel's gradient |
| `conv2dBackwardParametersPartial` + `reduceBatch` | one weight's gradient over one sample, then summed over batch |
| `maxPoolForward` / `maxPoolBackward` | one pooling window |
| `denseForward`, `denseBackwardParameters`, `denseBackwardInput` | one output, one weight gradient, one input gradient |
| `softmaxCrossEntropy` | one sample's loss and logit gradients |
| `sgdUpdate`, `momentumUpdate`, `adamUpdate` | one weight |

Two places differ from CPU loops, both to avoid many threads adding into
same memory:

- **Convolution input gradient.** CPU *scatters*: each output gradient is
  added to every input pixel under its window. On GPU that would need atomic
  adds. Instead each thread takes one input pixel and *gathers* from output
  positions whose window covered it: `oh = (ih + padding - kh) / stride`, when
  that divides evenly and is in range.
- **Convolution weight gradient.** One thread per weight summing over whole
  batch would use only 80 threads for first layer and leave GPU nearly
  idle. Sum is split in two: one thread per (sample, weight), then one per
  weight summing across batch.

Max pooling's backward pass does use `atomicAdd`, because overlapping windows
(stride < kernel) can share a winner; with 2x2 windows and stride 2 additions 
never collide.

### Accuracy

GPU computes in float, sums in a different order than CPU, and fuses
multiplies and adds (FMA). Its results therefore match CPU `ConvNet<float>`
to within about 1e-4 relative, not bit for bit. Unit tests check this for
logits, gradients and optimizer steps. Training curves on MNIST are
indistinguishable (loss 0.2251 after one epoch with every strategy).

### Building

`VisionModels.Cuda` project compiles `.cu` files through `nvcc-build.cmd`,
which sets up MSVC v143 environment for nvcc (CUDA 13 does not accept
MSVC 19.50+ as its host compiler) and targets compute capability 89 by default:

```bash
msbuild VisionModels.sln /p:Configuration=Release /p:Platform=x64 /p:CudaComputeCapability=86
```

Without CUDA Toolkit, project still is able to compile `CudaUnavailable.cpp` and 
`GpuConvNetUnavailable.cpp`. So everything links and `Cuda` reports that 
it is not available.

### Where Time Goes, and What Would Make It Faster

One epoch takes about 0.47 s, roughly 0.5 ms per step of 64 images. At this
size GPU is mostly waiting on kernel launches (about 35 per step: 9 forward,
17 backward, 1 gather, 8 optimizer) and per-step loss copy, not on arithmetic. 

Things to try:
- Launch with CUDA Graphs, or fuse Conv2D + ReLU and Dense + ReLU;
- Larger batches (e.g. `--batch 256`);
- Shared-memory tiling in convolution and dense kernels, or im2col plus a
  tiled matrix multiply;
- Profile it with Nsight Compute to see which kernels dominate.

## Getting Started

### 1. Get MNIST

```bash
powershell -ExecutionPolicy Bypass -File scripts/download_mnist.ps1
```

(or `sh scripts/download_mnist.sh` / `make mnist` on Linux). About 11 MB,
unpacked into `resources/mnist/`, which git ignores. Without it example
falls back to generated seven-segment digits (`--synthetic`).

### 2. Build

**Visual Studio 2026** (toolsets v145, plus v143 for CUDA project): open
`VisionModels.sln`, pick **Release | x64**, build. NuGet restores Google Test for
test project. If CUDA Toolkit is installed (`CUDA_PATH` set), GPU
code is compiled with nvcc for compute capability 8.9; use
`/p:CudaComputeCapability=86` (etc.) for another GPU. Without toolkit
everything still builds, and `Cuda` reports that it is not available.

**Linux / WSL** (g++ 13 or newer, `libgtest-dev` for tests):

```bash
make
make test
make CUDA=1 CUDA_ARCH=89
```

### 3. Run

```bash
bin/Release/x64/VisionModels.Examples.exe --strategy cuda --epochs 10
```

| Option | Meaning |
|---|---|
| `--strategy S` | `sequential`, `parallel` (default), `cuda`, or `all` to train same model with each and compare |
| `--epochs N` | passes over training set (default 2) |
| `--batch N` | images per step (default 64) |
| `--limit N` | train on first N images only, e.g. to try Sequential quickly |
| `--lr X`, `--optimizer O` | learning rate (0.001) and `adam`, `momentum` or `sgd` |
| `--threads N` | threads for `parallel` (default: all hardware threads) |
| `--synthetic` | generated digits instead of MNIST |
| `--show N` | draw N misclassified test digits as text art |

It prints model summary, loss while training, then test accuracy,
confusion matrix and some mistakes.

## Repository Layout

```
VisionModels.sln
VisionModels/include/        header-only library
  common/                    Tensor (NCHW), Parameter, Optimizer, ThreadPool, ParallelFor, Validation
  layers/                    Layer interface, Conv2D, MaxPool2D, ReLU, Flatten, Dense, SoftmaxCrossEntropy
  models/                    ConvNet, DigitCnn
  data/                      ImageDataset, MnistLoader, SyntheticDigits
  metrics/                   ConfusionMatrix
  pipelines/                 ExecutionStrategy, ImageClassifierTrainer
  cuda/                      CudaRuntime.h, GpuConvNet.h (plain C++ interfaces to GPU code)
VisionModels.Cuda/           static library: CUDA kernels (.cu, built by nvcc) or CPU-only stand-ins
VisionModels.Examples/       Main.cpp: train on MNIST, compare strategies
VisionModels.Tests/          Google Test Unit Tests
scripts/download_mnist.*     fetch MNIST into resources/mnist
```

## Using Library

```cpp
#include "DigitCnn.h"
#include "ImageClassifierTrainer.h"
#include "MnistLoader.h"

ImageDataset training = MnistLoader::loadTraining("resources/mnist");
ImageDataset test = MnistLoader::loadTest("resources/mnist");

ConvNet<float> net = makeDigitCnn<float>();
TrainingOptions options;
options.epochs = 3;
options.strategy = ExecutionStrategy::Cuda;       // or Sequential / Parallel
options.optimizer = OptimizerSettings::adam(1e-3);

ImageClassifierTrainer<float> trainer(net, options, std::cout);
trainer.train(training, &test);
EvaluationReport report = trainer.evaluate(test);
std::cout << report.accuracy() << "\n" << report.confusion.toString();
```

Or build your own network and train it step by step:

```cpp
RandomEngine rng(42);
ConvNet<double> net({ 1, 28, 28 });
net.addConv2D(6, 5, 1, 2, rng).addReLU().addMaxPool2D(2, 2)
   .addConv2D(16, 5, 1, 0, rng).addReLU().addMaxPool2D(2, 2)
   .addFlatten().addDense(120, rng).addReLU().addDense(84, rng).addReLU().addDense(10, rng);

float loss = net.trainBatch(images, labels, OptimizerSettings::withMomentum(0.01), /*threads*/ 8);
```

`add...()` checks that each layer fits one before, so a wrong size is
reported when network is built, not during training.

## Model

`makeDigitCnn()` in `models/DigitCnn.h` builds a small LeNet-style network:

```
Input                                     [1, 28, 28]
Conv2D(1 -> 8, 3x3, stride 1, padding 1)  [8, 28, 28]   80 parameters
ReLU                                      [8, 28, 28]
MaxPool2D(2x2, stride 2)                  [8, 14, 14]
Conv2D(8 -> 16, 3x3, stride 1, padding 1) [16, 14, 14]  1168 parameters
ReLU                                      [16, 14, 14]
MaxPool2D(2x2, stride 2)                  [16, 7, 7]
Flatten                                   [784]
Dense(784 -> 64)                          [64]          50240 parameters
ReLU                                      [64]
Dense(64 -> 10)                           [10]          650 parameters
Total parameters: 52138
```

followed by softmax cross-entropy loss. Each header explains its layer:

| Idea | Header |
|---|---|
| Convolution: weight sharing, padding, stride, one bias per filter, He init | `layers/Conv2D.h` |
| Max pooling and routing its gradient to window's winner | `layers/MaxPool2D.h` |
| ReLU, Flatten, fully connected layer | `layers/ReLU.h`, `layers/Flatten.h`, `layers/Dense.h` |
| Softmax + cross-entropy, computed together for stability | `layers/SoftmaxCrossEntropy.h` |
| SGD, SGD with momentum, Adam | `common/Optimizer.h`, `common/Parameter.h` |
| Network: shape checking, forward, backward, training step | `models/ConvNet.h` |
| MNIST's IDX file format | `data/MnistLoader.h` |
| Accuracy, precision, recall, F1, confusion matrix | `metrics/ConfusionMatrix.h` |

## How Three Strategies Work

**Sequential.** Each layer is written as loops over "items", where an item owns
its outputs: one output feature map in convolution's forward pass, one filter's 
weight gradient, one sample's input gradient. Sequential runs all items 
on calling thread.

**Parallel.** Same loops, split into chunks on a thread pool
(`common/ParallelFor.h`). Each item sums its inputs in same fixed order
whichever thread runs it, so Parallel produces **bit-identical** weights and
losses (tests check this). Getting that property means never letting two
threads add into same number. For example, weight gradient is split
by filter, not by sample.

**Cuda.** `cuda::GpuConvNet` rebuilds network on GPU from
`ConvNet::specs()`. Weights, gradients, Adam state, every activation and whole 
training set stay in GPU memory, so a step sends up only 64 image indices
and gets back 64 losses. Each kernel gives one GPU thread one output number and
mirrors a CPU loop. Trainer copies weights back into `ConvNet` after
each epoch, so model is always usable on CPU. Results match CPU to
within float rounding. See [CUDA Implementation](#cuda-implementation).

## Tests

```bash
bin/Release/x64/VisionModels.Tests.exe
```

57 tests (about 10 s in Release), including:

- Finite-difference gradient checks for every layer (with padding, stride and
  overlapping pool windows) and for a whole network;
- Known forward values for convolution, pooling, dense and loss;
- Parallel == Sequential, bit for bit, for layers and for whole training runs;
- GPU vs CPU logits, gradients and SGD / momentum / Adam steps (skipped without a GPU);
- Network learning synthetic digits to over 90% on each strategy;
- IDX reader against hand-made files, including broken ones.

## Future Ideas

- **im2col + matrix multiply** for Conv2D, and tiled shared-memory kernels on GPU.
- **Batch normalization** (its shift makes conv bias redundant) and **dropout**.
- **Data augmentation**: small shifts and rotations of digits.
- **CIFAR-10** (32x32 color, 3 channels): model already takes any `[C, H, W]`.
- Global average pooling instead of big Dense layer (50k of 52k parameters).

## License

MIT, see [LICENSE](LICENSE). MNIST is by Yann LeCun, Corinna Cortes and
Christopher J.C. Burges.
