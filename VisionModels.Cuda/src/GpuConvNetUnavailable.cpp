// The implementation of GpuConvNet.h used when the build has no CUDA Toolkit.
// The constructor throws CudaError, so no other member can ever be reached;
// they are defined only so that everything links.

#include "GpuConvNet.h"

namespace cuda {

	class GpuConvNet::Impl {
	};

	GpuConvNet::GpuConvNet(const std::vector<LayerSpec>& /*layers*/, const std::vector<std::size_t>& /*inputShape*/,
		std::size_t /*maxBatch*/) {
		throw CudaError("GpuConvNet: This build has no CUDA support.");
	}

	GpuConvNet::~GpuConvNet() = default;
	GpuConvNet::GpuConvNet(GpuConvNet&&) noexcept = default;
	GpuConvNet& GpuConvNet::operator=(GpuConvNet&&) noexcept = default;

	std::size_t GpuConvNet::maxBatch() const { return 0; }
	std::size_t GpuConvNet::classCount() const { return 0; }
	void GpuConvNet::uploadParameters(const std::vector<std::vector<float>>&, std::size_t) {}
	void GpuConvNet::downloadParameters(std::vector<std::vector<float>>&) const {}
	void GpuConvNet::downloadGradients(std::vector<std::vector<float>>&) const {}
	void GpuConvNet::uploadTrainingSet(const std::vector<float>&, const std::vector<std::uint8_t>&) {}
	std::size_t GpuConvNet::trainingSetSize() const { return 0; }
	float GpuConvNet::computeGradients(const std::vector<float>&, const std::vector<std::size_t>&) { return 0.0f; }
	float GpuConvNet::computeGradientsResident(const std::vector<std::uint32_t>&) { return 0.0f; }
	void GpuConvNet::update(const OptimizerSettings&) {}
	std::size_t GpuConvNet::steps() const { return 0; }
	void GpuConvNet::logits(const std::vector<float>&, std::size_t, std::vector<float>&) {}

} // namespace cuda
