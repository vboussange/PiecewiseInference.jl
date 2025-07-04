"""
$(DocStringExtensions.README)
"""
module PiecewiseInference
    using OrdinaryDiffEq
    using Optimization
    using OptimizationOptimisers, OptimizationOptimJL
    using MLUtils
    using Requires
    using DocStringExtensions

    using ForwardDiff, Zygote

    using LinearAlgebra
    using LaTeXStrings
    using UnPack
    using Statistics
    using Distributions
    import Distributions:loglikelihood #overwritten

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
    include("models/armodel.jl")
    include("InferenceProblem.jl")
    include("InferenceResult.jl")
    include("utils.jl")
    include("piecewise_loss.jl")
    include("inference.jl")
    include("evidence.jl")

    Base.@deprecate piecewise_MLE(args...; kwargs...) inference(args...; kwargs...)

    export simulate, ModelParams, @AnalyticModel, @ODEModel, @ARModel, name, remake
    export get_p, get_u0, get_alg, get_tspan, get_kwargs, get_mp, get_dims, get_prob
    export InferenceProblem, get_p, get_p_bijector, get_u0_bijector, get_re, get_tspan, get_model, get_mp
    export ParamFun, InferenceResult, get_p_trained, forecast
    export piecewise_loss
    export loss_param_prior_from_dict
    export inference, piecewise_ML_indep_TS, iterative_inference, get_ranges

end # module
