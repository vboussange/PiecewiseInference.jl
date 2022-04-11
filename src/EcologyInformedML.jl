__precompile__(false)

module EcologyInformedML
    using OrdinaryDiffEq
    using DiffEqFlux
    using PyPlot

    using ForwardDiff
    using LinearAlgebra
    using LaTeXStrings
    using UnPack
    using Statistics

    # parametric function
    abstract type ParamFun{N} end
    import Base.length
    length(::ParamFun{N}) where N = N

    """ 
        ResultMLE(minloss, p_trained, pred, ranges, losses,θs)
    Container for ouputs of MLE
    """
    struct ResultMLE{M,P,Pr,R,L,T}
        minloss::M
        p_trained::P
        pred::Pr
        ranges::R
        losses::L
        θs::T
    end

    include("utils.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")
    include("plot_convergence.jl")

    export ForwardDiffSensitivity # from DiffEqFlux and DiffEqSensitivity
    export ParamFun, ResultMLE
    export AIC, AICc, AICc_TREE, moments!, moments, FIM_strouwen, FIM_yazdani, divisors
    export minibatch_loss
    export minibatch_MLE, recursive_minibatch_MLE
    export plot_convergence
end # module
