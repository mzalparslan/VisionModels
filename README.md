# VisionModels

A from-scratch **C++20, header-only** implementation of convolutional neural
networks (CNNs), trained to recognize handwritten digits (MNIST): the classic
first computer-vision task.

There are no dependencies and no framework. Tensors, convolution, pooling,
backpropagation and optimizers are all written by hand, and every gradient is
checked against finite differences in the test suite. Google Test is used for
the tests only.

The same network trains three ways (`ExecutionStrategy`):

| Strategy | How | One MNIST epoch (60,000 images) | Speedup | Test accuracy after 1 epoch |
|---|---|---|---|---|
| `Sequential` | one CPU thread | 94.4 s | 1x | 97.53% |
| `Parallel` | 16 CPU threads, results bit-identical to Sequential | 19.7 s | 4.8x | 97.53% |
| `Cuda` | NVIDIA GPU, everything resident in GPU memory | 0.48 s | 197x | 97.58% |

Release build, batch 64, Adam (lr 0.001), RTX 4060 Ti and a 16-thread CPU. Ten
epochs on the GPU (4.7 s) reach **98.82%** test accuracy.

> This is a learning project. The kernels are the direct ("naive") algorithms,
> written to be read next to the math, not to compete with cuDNN or oneDNN.

It is a sibling of the MachineLearningModels (classical ML) and LanguageModels
(RNN, transformer, BERT, GPT) repositories and uses the same layout.

## Contents

- [The model](#the-model)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Using the library](#using-the-library)
- [How the three strategies work](#how-the-three-strategies-work)
- [Tests](#tests)
- [Ideas for next steps](#ideas-for-next-steps)
- [License](#license)

## The model

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
| Max pooling and routing its gradient to the window's winner | `layers/MaxPool2D.h` |
| ReLU, Flatten, fully connected layer | `layers/ReLU.h`, `layers/Flatten.h`, `layers/Dense.h` |
| Softmax + cross-entropy, computed together for stability | `layers/SoftmaxCrossEntropy.h` |
| SGD, SGD with momentum, Adam | `common/Optimizer.h`, `common/Parameter.h` |
| The network: shape checking, forward, backward, training step | `models/ConvNet.h` |
| MNIST's IDX file format | `data/MnistLoader.h` |
| Accuracy, precision, recall, F1, confusion matrix | `metrics/ConfusionMatrix.h` |

## Repository layout

```
VisionModels.sln
VisionModels/include/        header-only library
  common/                    Tensor (NCHW), Parameter, Optimizer, ThreadPool, ParallelFor, Validation
  layers/                    Layer interface, Conv2D, MaxPool2D, ReLU, Flatten, Dense, SoftmaxCrossEntropy
  models/                    ConvNet, DigitCnn
  data/                      ImageDataset, MnistLoader, SyntheticDigits
  metrics/                   ConfusionMatrix
  pipelines/                 ExecutionStrategy, ImageClassifierTrainer
  cuda/                      CudaRuntime.h, GpuConvNet.h (plain C++ interfaces to the GPU code)
VisionModels.Cuda/           static library: CUDA kernels (.cu, built by nvcc) or CPU-only stand-ins
VisionModels.Examples/       Main.cpp: train on MNIST, compare strategies
VisionModels.Tests/          Google Test unit tests
scripts/download_mnist.*     fetch MNIST into resources/mnist
docs/CUDA.md                 how the GPU version works
```

## Getting started

### 1. Get MNIST

```bash
powershell -ExecutionPolicy Bypass -File scripts/download_mnist.ps1
```

(or `sh scripts/download_mnist.sh` / `make mnist` on Linux). About 11 MB,
unpacked into `resources/mnist/`, which git ignores. Without it the example
falls back to generated seven-segment digits (`--synthetic`).

### 2. Build

**Visual Studio 2026** (toolsets v145, plus v143 for the CUDA project): open
`VisionModels.sln`, pick **Release | x64**, build. NuGet restores Google Test for
the test project. If the CUDA Toolkit is installed (`CUDA_PATH` set), the GPU
code is compiled with nvcc for compute capability 8.9; use
`/p:CudaComputeCapability=86` (etc.) for another GPU. Without the toolkit
everything still builds, and `Cuda` reports that it is not available.

**Linux / WSL** (g++ 13 or newer, `libgtest-dev` for the tests):

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
| `--strategy S` | `sequential`, `parallel` (default), `cuda`, or `all` to train the same model with each and compare |
| `--epochs N` | passes over the training set (default 2) |
| `--batch N` | images per step (default 64) |
| `--limit N` | train on the first N images only, e.g. to try Sequential quickly |
| `--lr X`, `--optimizer O` | learning rate (0.001) and `adam`, `momentum` or `sgd` |
| `--threads N` | threads for `parallel` (default: all hardware threads) |
| `--synthetic` | generated digits instead of MNIST |
| `--show N` | draw N misclassified test digits as text art |

It prints the model summary, the loss while training, then the test accuracy,
the confusion matrix and some mistakes.

## Using the library

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

`add...()` checks that each layer fits the one before, so a wrong size is
reported when the network is built, not during training.

## How the three strategies work

**Sequential.** Each layer is written as loops over "items", where an item owns
its outputs: one output feature map in the convolution's forward pass, one
filter's weight gradient, one sample's input gradient. Sequential runs all the
items on the calling thread.

**Parallel.** The same loops, split into chunks on a thread pool
(`common/ParallelFor.h`). Each item sums its inputs in the same fixed order
whichever thread runs it, so Parallel produces **bit-identical** weights and
losses (the tests check this). Getting that property means never letting two
threads add into the same number. For example, the weight gradient is split
by filter, not by sample.

**Cuda.** `cuda::GpuConvNet` rebuilds the network on the GPU from
`ConvNet::specs()`. The weights, gradients, Adam state, every activation and the
whole training set stay in GPU memory, so a step sends up only 64 image indices
and gets back 64 losses. Each kernel gives one GPU thread one output number and
mirrors a CPU loop. The trainer copies the weights back into the `ConvNet` after
each epoch, so the model is always usable on the CPU. Results match the CPU to
within float rounding. See [docs/CUDA.md](docs/CUDA.md).

## Tests

```bash
bin/Release/x64/VisionModels.Tests.exe
```

57 tests (about 10 s in Release), including:

- finite-difference gradient checks for every layer (with padding, stride and
  overlapping pool windows) and for a whole network;
- known forward values for convolution, pooling, dense and loss;
- Parallel == Sequential, bit for bit, for layers and for whole training runs;
- GPU vs CPU logits, gradients and SGD / momentum / Adam steps (skipped without a GPU);
- the network learning synthetic digits to over 90% on each strategy;
- the IDX reader against hand-made files, including broken ones.

## Ideas for next steps

- **im2col + matrix multiply** for Conv2D, and tiled shared-memory kernels on the GPU.
- **Batch normalization** (its shift makes the conv bias redundant) and **dropout**.
- **Data augmentation**: small shifts and rotations of the digits.
- **CIFAR-10** (32x32 color, 3 channels): the model already takes any `[C, H, W]`.
- Global average pooling instead of the big Dense layer (50k of the 52k parameters).

## License

MIT, see [LICENSE](LICENSE). MNIST is by Yann LeCun, Corinna Cortes and
Christopher J.C. Burges.
