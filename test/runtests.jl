using PiecewiseInference, Test, ForwardDiff, OrdinaryDiffEq
using SciMLSensitivity:ForwardDiffSensitivity
using OptimizationOptimJL:BFGS
using OptimizationOptimisers:Adam
using ParametricModels
using Distributions

@testset "PiecewiseInference" begin
    # include("FIM.jl")
    include("utils.jl")
    include("InferenceResult.jl")
    include("piecewise_loss.jl")
    include("inference.jl")
    # include("statistics.jl")
    include("evidence.jl")
end

if false # testing plot recipes
    @testset "PiecewiseInference" begin
        using PyPlot  # to test plot_convergence
        include("plot_convergence.jl")
    end
end