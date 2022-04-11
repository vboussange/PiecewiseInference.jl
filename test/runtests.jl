using EcologyInformedML, Test, ForwardDiff, PyPlot
using Revise

@testset "EcologyInformedML" begin
    # include("FIM.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")
end