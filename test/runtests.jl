using EcologyInformedML, Test, ForwardDiff, PyPlot, OrdinaryDiffEq
using DiffEqSensitivity: ForwardDiffSensitivity
using Revise

@testset "EcologyInformedML" begin
    # include("FIM.jl")
    # include("plot_convergence.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")
end