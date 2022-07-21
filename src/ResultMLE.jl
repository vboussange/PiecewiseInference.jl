""" 
$(SIGNATURES)

Container for ouputs of MLE.

# Notes
`res = ResultMLE()` has all fields empty but `res.minloss` which is set to `Inf`.

"""
Base.@kwdef struct ResultMLE{M,P,U0,Pr,R,L}
    minloss::M = Inf
    p_trained::P = []
    u0s_trained::U0 = [] #  initial condition vector estimated`[u_0_1, ..., u_0_n]`
                 # In the case of independent time series, 
                 # `[[u_0_TS1_1, ..., u_0_TS1_n],...,[u_0_TS1_1,...]]`, matching the format of `res.pred`.
    pred::Pr = []
    ranges::R = []
    losses::L = []
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
` loglike_fn = loglikelihood_lognormal`. In such case, Σ corresponds to the standard deviation of the log normal noise.
"""
function loglikelihood(res::ResultMLE, data_set::Array, Σ; loglike_fn = loglikelihood_normal) 
    isempty(res.pred) ? error("`res.pred` should not be empty, use `minibatch_MLE` with `save_pred = true`") : nothing
    return loglikelihood(res.pred, res.ranges, data_set, Σ, loglike_fn)
end

#specialised version for minibatch
function loglikelihood(pred::Vector, ranges::Vector, data_set::Array, Σ, loglike_fn) 
    if typeof(pred) <: Vector{Vector{Array{T}}} where T # for independent time series
        error("Function to yet implemented from `ResultMLE` with Independent time series")
    else
        pred_all_batches = cat( [pred[i] for (i,rng) in enumerate(ranges)]..., dims=2)
        data_all_batches = cat( [data_set[:,rng] for (i,rng) in enumerate(ranges)]..., dims=2)
        logl = loglike_fn(pred_all_batches, data_all_batches, Σ)
    end
    return logl
end
# see https://juliaeconomics.com/2014/06/16/numerical-maximum-likelihood-the-ols-example/


function loglikelihood_normal(pred_all_batches::Array, data_all_batches::Array, Σ::Array)
    return sum([logpdf(MvNormal(pred_all_batches[:,i], Σ), data_all_batches[:,i]) for i in 1:size(pred_all_batches,2)])
end

function loglikelihood_lognormal(pred_all_batches::Array, data_all_batches::Array, σ::Number)
    l = 0.
    for i in 1:size(pred_all_batches,2)
        for j in 1:size(pred_all_batches,1)
            if pred_all_batches[j,i] > 0. && data_all_batches[j,i] > 0.
                l += logpdf(LogNormal(log.(pred_all_batches[j,i]), σ), data_all_batches[j,i])
            end
        end
    end
    return l
    # NOTE: pdf(MvNormal(zeros(2), σ^2 * diagm(ones(2))), [0.,0.]) ≈ pdf(Normal(0.,σ),0.)^2
end

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