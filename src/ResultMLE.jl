""" 
$(SIGNATURES)

Container for ouputs of MLE.

# Notes
`res = ResultMLE()` has all fields empty but `res.minloss` which is set to `Inf`.

"""
Base.@kwdef struct ResultMLE{M,P,Pp,Pl,Pr,R,L,T}
    minloss::M = Inf
    p_trained::P = []
    p_true::Pp = []
    p_labs::Pl = []
    pred::Pr = []
    ranges::R = []
    losses::L = []
    θs::T = []
end

"""
$(SIGNATURES)

Returns initial condition vector estimated`[u_0_1, ..., u_0_n]`
, where `n` corresponds to the number of chunks.
In the case of independent time series, returns 
`[[u_0_TS1_1, ..., u_0_TS1_n],...,[u_0_TS1_1,...]]`, matching the format of `res.pred`.

"""
function get_u0s(res::ResultMLE)
    u0s_init = similar(res.pred)
    if typeof(res.pred) <: Vector{Vector{Array{T}}} where T
        # result obtained from fn `minibatch_ML_indep_TS`
        for (i,TS) in enumerate(res.pred)
            u0s_init[i] = [chunk[:,1] for chunk in TS]
        end
    else
        u0s_init = [chunk[:,1] for chunk in res.pred]
    end
    return u0s_init
end

"""
$(SIGNATURES)

Computes the RSS of `res` given `data_set`.
# Argument:
- `fn` is set to `identity` but can be e.g. `log` if lognormal loss used.
"""
function RSS(res::ResultMLE, data_set::Array, fn = identity)
    if typeof(res.pred) <: Vector{Vector{Array{T}}} where T
        error("not yet implemented for independent time series")
    else
        # version where we reomve the additional time point used for multiple shooting
        # ???
        # version where the additional time point is kept
        ϵ = cat( [fn.(res.pred[i]) .- fn.(data_set[:,rng]) for (i,rng) in enumerate(res.ranges)]..., dims=2)
        rss = sum(ϵ.^2) 
    end
    return rss
end

"""
$(SIGNATURES)

Computes the loglikelihood of `res` given the observational noise variance covariance matrix Σ.

# Options
By default, normal observational noise is assumed, 
but lognormal observational noise can be chosen by setting
`distrib=MvLogNormal, fn=log`
"""
function loglikelihood(res::ResultMLE, data_set::Array, Σ::Array; distrib=MvNormal, fn=identity)
    isempty(res.pred) ? error("`res.pred` should not be empty, use `minibatch_MLE` with `save_pred = true`") : nothing
    if typeof(res.pred) <: Vector{Vector{Array{T}}} where T
        dim_prob = size(data_set[1],1)
        nb_ts = length(res.ranges)
        # looping over time series
        logl = 0.
        for k in 1:nb_ts
            data_set_simu_vect_k = [res.pred[k][i][:,1:end-1] for i in 1:length(res.ranges[k])-1] # removing all duplicates shared by segments i, i-1, removing the last one
            push!(data_set_simu_vect_k,res.pred[k][end])
            pred_k = cat(data_set_simu_vect_k..., dims=2)
            data_set_k = data_set[k]
            logl += sum(logpdf(distrib(fn.(pred_k[:,i]), Σ), data_set_k[:,i]) for i in 1:length(pred_k) ) 
        end
    else
        # version where we reomve the additional time point used for multiple shooting
        # dim_prob = size(data_set,1)
        # pred_vect = [res.pred[i][:,1:end-1] for i in 1:length(res.ranges)-1] # removing all duplicates shared by segments i, i-1, removing the last one
        # push!(pred_vect,res.pred[end])
        # pred = cat(pred_vect..., dims=2)
        # ϵ = (data_set - pred)
        # logl = sum(log(pdf(MvNormal(zeros(dim_prob), Σ), ϵ_i)) for ϵ_i in eachcol(ϵ) ) 

        # version where the additional time point is kept
        dim_prob = size(data_set,1)
        ϵ = cat( [res.pred[i] .- data_set[:,rng] for (i,rng) in enumerate(res.ranges)]..., dims=2)
        logl = sum(logpdf(MvNormal(zeros(dim_prob), Σ), ϵ_i) for ϵ_i in eachcol(ϵ) ) 
    end
    return logl
end
# see https://juliaeconomics.com/2014/06/16/numerical-maximum-likelihood-the-ols-example/

"""
$(SIGNATURES)

Computes the AIC of `res` given the observational noise variance covariance matrix Σ.
"""
function AIC(res::ResultMLE, data_set::Array, Σ::Array)
    nparams = length(res.p_trained)
    logl = loglikelihood(res, data_set, Σ)
    AIC_likelihood = - 2 * logl + 2 * nparams

    return AIC_likelihood
end

"""
$(SIGNATURES)

Computes the AIC given loglikelihood `logl` and number of parameters `nparams`.
"""
function AIC(logl::T, nparams::P) where {T <: AbstractFloat, P <: Integer}
    AIC_likelihood = - 2 * logl + 2 * nparams
    return AIC_likelihood
end