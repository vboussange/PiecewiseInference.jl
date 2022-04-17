__precompile__(false)

module EcologyInformedML
    using OrdinaryDiffEq
    using DiffEqFlux
    using Requires

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
        ResultMLE()
    Container for ouputs of MLE.
    
    # Notes
    `res = ResultMLE()` has all fields empty but `res.minloss` which is set to `Inf`.

    """
    struct ResultMLE{M,P,Pp,Pl,Pr,R,L,T}
        minloss::M
        p_trained::P
        p_true::Pp
        p_labs::Pl
        pred::Pr
        ranges::R
        losses::L
        θs::T
    end
    function ResultMLE()
        ResultMLE(Inf, [], [], [], [], [], [], [])
    end

    include("utils.jl")
    include("minibatch_loss.jl")
    include("minibatch_MLE.jl")

    plot_convergence(args...;kwargs...) = println("Plotting requires loading package `PyPlot`")
    function __init__()
        @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" include("plot_convergence.jl")
    end

    export ForwardDiffSensitivity # from DiffEqFlux and DiffEqSensitivity
    export ParamFun, ResultMLE
    export AIC, AICc, AICc_TREE, moments!, moments, FIM_strouwen, FIM_yazdani, divisors
    export minibatch_loss
    export minibatch_MLE, minibatch_ML_indep_TS, iterative_minibatch_MLE
    export plot_convergence
end # module
