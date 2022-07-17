using MiniBatchInference, Test, ForwardDiff, OrdinaryDiffEq
using DiffEqFlux:group_ranges,ADAM,BFGS,ForwardDiffSensitivity
using Revise

@testset "MiniBatchInference" begin
    # include("FIM.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")
end

if false # testing plot recipes
    @testset "MiniBatchInference" begin
        using PyPlot  # to test plot_convergence
        include("plot_convergence.jl")
    end
end