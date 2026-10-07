CXX ?= g++

BUILD ?= release
BUILD_DIR := build/$(BUILD)
OBJECT_DIR := $(BUILD_DIR)/obj
BINARY_DIR := $(BUILD_DIR)/bin

EXAMPLE_NAME := $(BINARY_DIR)/VisionModels.Examples
TEST_NAME := $(BINARY_DIR)/VisionModels.Tests

# VisionModels itself is header-only (every implementation file is a class
# template), so there's no library source to compile, only the include paths.
INCLUDE_DIRS := \
	-IVisionModels/include/common \
	-IVisionModels/include/layers \
	-IVisionModels/include/models \
	-IVisionModels/include/data \
	-IVisionModels/include/metrics \
	-IVisionModels/include/pipelines \
	-IVisionModels/include/cuda \
	-IVisionModels.Tests

CPPFLAGS := $(INCLUDE_DIRS) -MMD -MP
COMMON_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic -pthread

# Release by default: training in -O0 is many times slower.
ifeq ($(BUILD),release)
	CXXFLAGS := $(COMMON_FLAGS) -O3 -DNDEBUG
else
	CXXFLAGS := $(COMMON_FLAGS) -O0 -g
endif

EXAMPLE_SOURCES := $(wildcard VisionModels.Examples/*.cpp)
EXAMPLE_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(EXAMPLE_SOURCES))

# pch.cpp only builds Visual Studio's precompiled header.
TEST_SOURCES := $(shell find VisionModels.Tests -type f -name '*.cpp' ! -name pch.cpp)
TEST_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(TEST_SOURCES))

# Google Test (e.g. `apt install libgtest-dev`).
GTEST_LIBS ?= -lgtest_main -lgtest -pthread

# CUDA (optional): `make CUDA=1` compiles the GPU code with nvcc and links the CUDA
# runtime; without it a stand-in reports CUDA as unavailable and everything else
# builds and runs. CUDA_ARCH is the GPU's compute capability without the dot
# (89 = RTX 40 series).
CUDA ?= 0
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= 89
NVCC ?= $(CUDA_PATH)/bin/nvcc
ifeq ($(CUDA),1)
	CUDA_OBJECTS := $(OBJECT_DIR)/VisionModels.Cuda/src/CudaRuntime.o \
		$(OBJECT_DIR)/VisionModels.Cuda/src/GpuConvNet.o
	CUDA_LIBS := -L$(CUDA_PATH)/lib64 -lcudart
else
	CUDA_OBJECTS := $(OBJECT_DIR)/VisionModels.Cuda/src/CudaUnavailable.o \
		$(OBJECT_DIR)/VisionModels.Cuda/src/GpuConvNetUnavailable.o
	CUDA_LIBS :=
endif

DEPENDENCY_FILES := $(EXAMPLE_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d) $(CUDA_OBJECTS:.o=.d)

.PHONY: all examples tests test run mnist clean help

all: examples

examples: $(EXAMPLE_NAME)

tests: $(TEST_NAME)

test: $(TEST_NAME)
	$(TEST_NAME)

run: examples
	$(EXAMPLE_NAME) $(ARGS)

mnist:
	sh scripts/download_mnist.sh

$(EXAMPLE_NAME): $(EXAMPLE_OBJECTS) $(CUDA_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(CUDA_LIBS) -o $@

$(TEST_NAME): $(TEST_OBJECTS) $(CUDA_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(GTEST_LIBS) $(CUDA_LIBS) -o $@

# The tests include "pch.h" (Google Test) as Visual Studio needs; here it is an
# ordinary header found through -IVisionModels.Tests.
$(OBJECT_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJECT_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(INCLUDE_DIRS) -std=c++20 -O2 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -c $< -o $@

clean:
	rm -rf build

help:
	@echo "make                  Build the example executable (release)"
	@echo "make BUILD=debug      Build without optimization, with debug info"
	@echo "make run ARGS='--strategy all --limit 10000'   Build and run it"
	@echo "make tests            Build the Google Test executable"
	@echo "make test             Build and run all tests"
	@echo "make CUDA=1           Build with the GPU code (needs the CUDA Toolkit); also for tests and run"
	@echo "make mnist            Download MNIST into resources/mnist"
	@echo "make clean            Remove Makefile build outputs"

-include $(DEPENDENCY_FILES)
