#!/bin/sh
# Downloads the MNIST handwritten digit dataset into resources/mnist and unpacks it.
#
# Usage (from the repository root): sh scripts/download_mnist.sh
#
# The files come from the mirror torchvision uses; yann.lecun.com, the
# original home of MNIST, often refuses downloads now.
set -e
mirror=https://ossci-datasets.s3.amazonaws.com/mnist
target="$(dirname "$0")/../resources/mnist"
mkdir -p "$target"
for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ -f "$target/$name" ]; then
        echo "$name already there"
        continue
    fi
    echo "Downloading $name.gz"
    curl -fL -o "$target/$name.gz" "$mirror/$name.gz"
    gunzip -f "$target/$name.gz"
done
echo "MNIST is in $target"
