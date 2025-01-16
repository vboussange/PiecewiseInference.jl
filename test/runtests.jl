using PiecewiseInference, Test, ForwardDiff, OrdinaryDiffEq
using SciMLSensitivity:ForwardDiffSensitivity
using OptimizationOptimJL:BFGS
using OptimizationOptimisers:Adam
using Distributions

@testset "PiecewiseInference" begin
    # include("FIM.jl")
    include("models/analyticmodel.jl")
    include("models/odemodel.jl")
    include("models/armodel.jl")

    include("utils.jl")
    include("InferenceResult.jl")
    include("piecewise_loss.jl")
    include("inference.jl")
    # include("statistics.jl")
end