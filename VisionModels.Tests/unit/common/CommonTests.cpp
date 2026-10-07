#include "pch.h"

#include <cmath>

#include "ExecutionStrategy.h"
#include "Parameter.h"
#include "Tensor.h"

// ------------------------------------------------------------------- Tensor

TEST(TensorTests, ShapeSizeAndReshape) {
	Tensor<float> tensor({ 2, 3, 4, 5 }, 1.5f);
	EXPECT_EQ(tensor.size(), 120u);
	EXPECT_EQ(tensor.rank(), 4u);
	tensor.reshape({ 2, 60 });
	EXPECT_EQ(tensor.shape, (std::vector<std::size_t>{ 2, 60 }));
	EXPECT_THROW(tensor.reshape({ 7, 7 }), InvalidSizeError);
	EXPECT_THROW(Tensor<float>({ std::size_t(-1), 4 }), InvalidSizeError);
}

// ---------------------------------------------------------------- Parameter

namespace {

	Parameter<double> makeParameter(double value, double gradient) {
		Parameter<double> parameter;
		parameter.initConstant({ 1 }, value);
		parameter.grad[0] = gradient;
		return parameter;
	}

}

TEST(ParameterTests, SgdStepsAgainstTheGradient) {
	Parameter<double> parameter = makeParameter(1.0, 0.5);
	parameter.update(OptimizerSettings::sgd(0.1), 1);
	EXPECT_DOUBLE_EQ(parameter.value[0], 0.95);
}

TEST(ParameterTests, MomentumAccumulatesVelocity) {
	Parameter<double> parameter = makeParameter(1.0, 1.0);
	const OptimizerSettings settings = OptimizerSettings::withMomentum(0.1, 0.9);
	parameter.update(settings, 1); // v = 1,   w = 0.9
	parameter.update(settings, 2); // v = 1.9, w = 0.71
	EXPECT_NEAR(parameter.value[0], 0.71, 1e-12);
}

TEST(ParameterTests, AdamFirstStepMovesByLearningRate) {
	// On step 1 the bias-corrected m and v are g and g^2, so the step is
	// lr * g / (|g| + eps): about lr in the direction against the gradient.
	Parameter<double> parameter = makeParameter(1.0, 123.0);
	parameter.update(OptimizerSettings::adam(0.01), 1);
	EXPECT_NEAR(parameter.value[0], 0.99, 1e-9);
}

TEST(ParameterTests, RejectsBadSettingsAndValues) {
	Parameter<double> parameter = makeParameter(1.0, 1.0);
	EXPECT_THROW(parameter.update(OptimizerSettings::sgd(0.0), 1), InvalidParameterError);
	EXPECT_THROW(parameter.update(OptimizerSettings::adam(0.01), 0), InvalidParameterError);
	OptimizerSettings badMomentum = OptimizerSettings::withMomentum(0.1, 1.0);
	EXPECT_THROW(parameter.update(badMomentum, 1), InvalidParameterError);
	parameter.grad[0] = std::nan("");
	EXPECT_THROW(parameter.update(OptimizerSettings::sgd(0.1), 1), NaNError);
}

TEST(ParameterTests, InitNormalIsReproducible) {
	RandomEngine first(5);
	RandomEngine second(5);
	Parameter<float> a;
	Parameter<float> b;
	a.initNormal({ 4, 4 }, 0.1f, first);
	b.initNormal({ 4, 4 }, 0.1f, second);
	EXPECT_EQ(a.value.data, b.value.data);
	EXPECT_THROW(a.initNormal({ 4, 0 }, 0.1f, first), InvalidParameterSizeError);
}

// -------------------------------------------------------- ExecutionStrategy

TEST(ExecutionStrategyTests, ParsesAndNames) {
	EXPECT_EQ(parseExecutionStrategy("sequential"), ExecutionStrategy::Sequential);
	EXPECT_EQ(parseExecutionStrategy("parallel"), ExecutionStrategy::Parallel);
	EXPECT_EQ(parseExecutionStrategy("cuda"), ExecutionStrategy::Cuda);
	EXPECT_THROW(parseExecutionStrategy("gpu"), InvalidParameterError);
	EXPECT_STREQ(toString(ExecutionStrategy::Cuda), "Cuda");
}
