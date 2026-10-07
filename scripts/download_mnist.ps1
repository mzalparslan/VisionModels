# Downloads the MNIST handwritten digit dataset into resources/mnist and unpacks it.
#
# Usage (from the repository root):
#   powershell -ExecutionPolicy Bypass -File scripts/download_mnist.ps1
#
# The files come from the mirror torchvision uses; yann.lecun.com, the
# original home of MNIST, often refuses downloads now. Four .gz files,
# about 11 MB in total, 55 MB unpacked.

$ErrorActionPreference = 'Stop'
$mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
$target = Join-Path $PSScriptRoot '..\resources\mnist'
New-Item -ItemType Directory -Force $target | Out-Null

foreach ($name in 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte') {
    $file = Join-Path $target $name
    if (Test-Path $file) {
        Write-Host "$name already there"
        continue
    }
    $archive = "$file.gz"
    Write-Host "Downloading $name.gz"
    Invoke-WebRequest -Uri "$mirror/$name.gz" -OutFile $archive

    # Unpack with .NET's GZipStream: no extra tools needed.
    $source = [System.IO.File]::OpenRead($archive)
    $destination = [System.IO.File]::Create($file)
    $gzip = New-Object System.IO.Compression.GZipStream($source, [System.IO.Compression.CompressionMode]::Decompress)
    $gzip.CopyTo($destination)
    $gzip.Dispose()
    $destination.Dispose()
    $source.Dispose()
    Remove-Item $archive
}
Write-Host "MNIST is in $((Resolve-Path $target).Path)"
