__precompile__(false)

module MiniBatchInference
    using OrdinaryDiffEq
    using DiffEqFlux
    using Requires
    using DocStringExtensions

    using ForwardDiff
    using LinearAlgebra
    using LaTeXStrings
    using UnPack
    using Statistics, Distributions

    # parametric function
    abstract type ParamFun{N} end
    import Base.length
    length(::ParamFun{N}) where N = N

    include("ResultMLE.jl")
    include("utils.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")

    plot_convergence(args...;kwargs...) = println("Plotting requires loading package `PyPlot`")
    function __init__()
        @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" include("plot_convergence.jl")
    end

    export ForwardDiffSensitivity # from DiffEqFlux and DiffEqSensitivity
    export ParamFun, ResultMLE, get_u0s
    export AIC, AICc, AICc_TREE, moments!, moments, FIM_strouwen, FIM_yazdani, divisors, loglikelihood, RSS
    export minibatch_loss
    export minibatch_MLE, minibatch_ML_indep_TS, iterative_minibatch_MLE
    export plot_convergence
end # module
