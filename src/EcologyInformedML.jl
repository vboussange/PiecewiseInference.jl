module EcologyInformedML

    using Reexport

    @reexport using OrdinaryDiffEq
    @reexport using DiffEqFlux
    @reexport using PyPlot

    using ForwardDiff
    using LinearAlgebra
    using LaTeXStrings
    using UnPack
    using Statistics

    # parametric function
    abstract type ParamFun{N} end
    import Base.length
    length(::ParamFun{N}) where N = N

    include("utils.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")
    include("plot_convergence.jl")

    export AIC, AICc, AICc_TREE, moments!, moments, FIM_strouwen, FIM_yazdani, divisors
    export minibatch_loss
    export minibatch_MLE, recursive_minibatch_MLE
    export plot_convergence
end # module
