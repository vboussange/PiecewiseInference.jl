
"""
$(SIGNATURES)

Computes the RSS of `res` given `data_set`.
# Argument:
- `fn` is set to `identity` but can be e.g. `log` if lognormal loss used.
"""
function RSS(res::ResultMLE, data_set::Array, noisedistrib::T)
    if typeof(noisedistrib) isa MvNormal
        fn = identity
    elseif typeof(noisedistrib) isa MvLogNormal
        fn = log
    else
        error("RSS for $T distributed eror is not defined")
    end
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

    Computes the loglikelihood of `res` given the observational noise variance distribution `noisedistrib`.

# Args
- `noisedistrib` corresponds to the assumed distribution of the noise. 
It can be `:MvLogNormal` or `:MvNormal` (comprising the multivariate types)
"""
# loglikelihood `ResultMLE` instead of `InferenceResult` cannot simulate model because it does not store it.
# Hence it needs to store the predictions.
# This should be fixed in the future
function loglikelihood(res::ResultMLE, data_set::Array, noisedistrib::Sampleable)
    isempty(res.pred) ? error("`res.pred` should not be empty, use `piecewise_MLE` with `save_pred = true`") : nothing
    return loglikelihood(res.pred, res.ranges, data_set, noisedistrib)
end

#specialised version for piecewise
function loglikelihood(pred::Vector, ranges::Vector, data_set::Array, noisedistrib) 
    if typeof(pred) <: Vector{Vector{Array{T}}} where T # for independent time series
        error("Function to yet implemented from `ResultMLE` with Independent time series")
    else
        pred_all_batches = cat( [pred[i] for (i,rng) in enumerate(ranges)]..., dims=2)
        data_all_batches = cat( [data_set[:,rng] for (i,rng) in enumerate(ranges)]..., dims=2)
        logl = loglikelihood(pred_all_batches, data_all_batches, noisedistrib)
    end
    return logl
end
# see https://juliaeconomics.com/2014/06/16/numerical-maximum-likelihood-the-ols-example/


function loglikelihood(pred_all_batches::Array, data_all_batches::Array, noisedistrib::MvNormal)
    @assert all(noisedistrib.μ .== 0.) "`noisedistrib` must have 0 mean, because the mean error should be zero"
    l = 0.
    for i in 1:size(pred_all_batches,2)
        ϵ = pred_all_batches[:,i] - data_all_batches[:,i]
        l += logpdf(noisedistrib, ϵ)
    end
    return l
end

get_μ(dist::MvLogNormal) = dist.normal.μ

function loglikelihood(pred_all_batches::Array, data_all_batches::Array, noisedistrib::MvLogNormal)
    l = 0.
    @assert all(get_μ(noisedistrib) .== 0.) "`noisedistrib` must have 0 mean, because the mean error should be zero"
    for i in 1:size(pred_all_batches,2)
        if all(pred_all_batches[:,i] .> 0.) && all(data_all_batches[:,i] .> 0.)
            ϵ = log.(pred_all_batches[:,i]) - log.(data_all_batches[:,i])
            l += logpdf(noisedistrib.normal, ϵ) # see Schartau 2017 (https://bg.copernicus.org/articles/14/1647/2017/) Eq. 14.
        end
    end
    return l - sum(log.(pred_all_batches))
    # NOTE: pdf(MvNormal(zeros(2), σ^2 * diagm(ones(2))), [0.,0.]) ≈ pdf(MvNormal(0.,σ),0.)^2
end

function loglikelihood(res::InferenceResult, 
                        ode_data::Array, 
                        noisedistrib;
                        u0s = res.res.u0s_trained,
                        p = res.res.p_trained) # we take res.res.p_trained because we would have to transform the parameters otherwise
    p, _ = Optimisers.destructure(p)
    p = p |> res.m.mp.st
    θ = [u0s...;p] 
    loss_fn(data, params, pred, rg) = PiecewiseInference.loglikelihood(pred, data, noisedistrib)
    l, _ = piecewise_loss(θ, ode_data, get_kwargs(res.m)[:saveat], res.m, loss_fn, res.res.ranges; continuity_term=0.)
    return l
end

"""
$(SIGNATURES)

Computes the AIC of `res` given the observational noise distribution `noisedistrib`.
"""
function AIC(res::ResultMLE, data_set::Array, noisedistrib::Sampleable)
    nparams = length(res.p_trained)
    logl = loglikelihood(res, data_set, noisedistrib)
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


"""
$(SIGNATURES)

Estimate noise variance, assuming similar noise across all the dimensions of the data.
"""
function estimate_σ(pred::Array, odedata::Array, ::MvNormal)
    @assert size(pred) == size(odedata)
    RSS = (pred .- odedata).^2
    return sqrt(mean(RSS))
end

function estimate_σ(pred::Array, odedata::Array, ::MvLogNormal)
    @assert size(pred) == size(odedata)
    logsquare = (log.(pred) .- log.(odedata)).^2
    return sqrt(mean(logsquare[logsquare .< Inf]))
end

function estimate_σ(reseco::InferenceResult, odedata::Array; noisedistrib=MvLogNormal(), include_ic = true)
    @assert !isempty(reseco.res.pred) "reseco should have been obtained with `save_pred = true`"
    if include_ic
    odedata = hcat([odedata[:,rg] for rg in reseco.res.ranges]...)
    pred = hcat([reseco.res.pred[i] for i in 1:length(reseco.res.ranges)]...)
    else
    odedata = hcat([odedata[:,rg[2:end]] for rg in reseco.res.ranges]...)
    pred = hcat([reseco.res.pred[i][:,2:end] for i in 1:length(reseco.res.ranges)]...)
    end
    return estimate_σ(pred, odedata, noisedistrib)
end


# TODO: test it !
"""
$(SIGNATURES)

"""
function get_var_covar_matrix(reseco::InferenceResult, odedata::Array, noisedistrib::Sampleable)
    likelihood_fn_optim(p) = Econobio.loglikelihood(reseco, odedata, noisedistrib; p = p)
    p_trained = reseco.res.p_trained
    numerical_hessian = ForwardDiff.hessian(likelihood_fn_optim, p_trained)
    var_cov_matrix = - inv(numerical_hessian)
    return var_cov_matrix
end

"""
$(SIGNATURES)

Compute confidence intervals, given `var_cov_matrix`, parameters `p` and a confidence level `α`.
"""
function compute_cis(var_cov_matrix::Matrix, p::Vector, α::AbstractFloat)
    ses = sqrt.(diag(var_cov_matrix)) 
    τ = cquantile(Normal(0, 1), α)
    lower = p - τ * ses
    upper = p + τ * ses
    lower, upper
end

"""
$(SIGNATURES)

"""
compute_cis(reseco::InferenceResult, odedata, noisedistrib, p, α, σ) = compute_cis(get_var_covar_matrix(reseco, 
                                                                                                        odedata,
                                                                                                        noisedistrib),
                                                                                                        p, α)


"""
$(SIGNATURES)

    We distinguish between R2 for log transformed values (with `dist::MvLogNormal` as last argument) 
    and standard R2, to discard non positive values in the former case.
"""
function R2(odedata::AbstractArray, pred::AbstractArray, ::MvLogNormal)
    rsstot = log.(odedata) .- mean(log.(odedata), dims=1)
    rssreg = log.(pred) .- log.(odedata)

    padding = (rsstot .< Inf) .* (rssreg .< Inf)

    vartot = sum(abs2,rsstot[padding])
    var_reg = sum(abs2, rssreg[padding])
    R2 = 1 - var_reg / vartot
    return R2
end

function R2(odedata::AbstractArray, pred::AbstractArray)
    rsstot = odedata .- mean(odedata, dims=1)
    rssreg = pred .- odedata

    vartot = sum(abs2,rsstot)
    var_reg = sum(abs2, rssreg)
    R2 = 1 - var_reg / vartot
    return R2
end

function R2(inf_res::InferenceResult, data_set::AbstractArray, args...) 
    pred = inf_res.res.pred
    ranges = inf_res.res.ranges
    if typeof(pred) <: Vector{Vector{Array{T}}} where T # for independent time series
        error("Function to yet implemented from `ResultMLE` with Independent time series")
    else
        pred_all_batches = cat( [pred[i] for (i,rng) in enumerate(ranges)]..., dims=2)
        data_all_batches = cat( [data_set[:,rng] for (i,rng) in enumerate(ranges)]..., dims=2)
        logl = R2(pred_all_batches, data_all_batches, args...)
    end
    return logl
end