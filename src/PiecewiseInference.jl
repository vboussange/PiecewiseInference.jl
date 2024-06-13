__precompile__()
"""
$(DocStringExtensions.README)
"""
module PiecewiseInference
    using OrdinaryDiffEq
    using Optimization
    using OptimizationOptimJL:Optim
    using Requires
    using DocStringExtensions

    using ForwardDiff
    using LinearAlgebra
    using LaTeXStrings
    using UnPack
    using Statistics
    using Distributions
    import Distributions:loglikelihood #overwritten

    using Optimisers, Flux
    using IterTools: ncycle 
    using Bijectors
    using SciMLBase
    using ComponentArrays
    using ChainRulesCore # used to ignore blocks of code

    # parametric function
    abstract type ParamFun{N} end
    import Base.length
    length(::ParamFun{N}) where N = N
    include("models/models.jl")
    include("models/odemodel.jl")
    include("models/analyticmodel.jl")
    include("InferenceProblem.jl")
    include("InferenceResult.jl")
    include("utils.jl")
    include("piecewise_loss.jl")
    include("inference.jl")
    include("statistics.jl")
    include("evidence.jl")

    Base.@deprecate piecewise_MLE(args...; kwargs...) inference(args...; kwargs...)

    plot_convergence(args...;kwargs...) = println("Plotting requires loading package `PyPlot`")
    function __init__()
        @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" include("plot_convergence.jl")
    end

    export simulate, ModelParams, @AnalyticModel, @ODEModel, name, remake
    export get_p, get_u0, get_alg, get_tspan, get_kwargs, get_mp, get_dims, get_prob
    export InferenceProblem, get_p, get_p_bijector, get_u0_bijector, get_re, get_tspan, get_model, get_mp
    export ParamFun, InferenceResult, get_p_trained, forecast
    export group_ranges, AIC, AICc, AICc_TREE, moments!, moments, divisors
    export piecewise_loss
    export inference, piecewise_ML_indep_TS, iterative_inference, get_ranges
    export plot_convergence
    export FIM_strouwen, FIM_yazdani, loglikelihood, estimate_σ, RSS, R2, pretty_print, loss_param_prior_from_dict, get_evidence,
            loglikelihood_lognormal
end # module
