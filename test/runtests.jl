using PiecewiseInference, Test, ForwardDiff, OrdinaryDiffEq
using DiffEqSensitivity:ForwardDiffSensitivity
using OptimizationOptimJL:BFGS
using OptimizationOptimisers:Adam
using ParametricModels
using Distributions

@testset "PiecewiseInference" begin
    # include("FIM.jl")
    include("ResultMLE.jl")
    include("piecewise_loss.jl")
    include("piecewise_MLE.jl")
    include("statistics.jl")
end

if false # testing plot recipes
    @testset "PiecewiseInference" begin
        using PyPlot  # to test plot_convergence
        include("plot_convergence.jl")
    end
end