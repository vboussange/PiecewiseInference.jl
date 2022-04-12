using EcologyInformedML, Test, ForwardDiff, OrdinaryDiffEq
using Revise

@testset "EcologyInformedML" begin
    # include("FIM.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")
end

if false # testing plot recipes
    @testset "EcologyInformedML" begin
        using PyPlot  # to test plot_convergence
        include("plot_convergence.jl")
    end
end