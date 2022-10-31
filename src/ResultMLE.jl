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
    isempty(res.pred) ? error("`res.pred` should not be empty, use `piecewise_MLE` with `save_pred = true`") : nothing
    return loglikelihood(res.pred, res.ranges, data_set, Σ, loglike_fn)
end

#specialised version for piecewise
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


function loglikelihood_normal(pred_all_batches::Array, data_all_batches::Array, Σ)
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


"""
$(SIGNATURES)

"""
Base.@kwdef struct InferenceResult{Model<:AbstractModel, RES}
    m::Model
    res::RES
end
import Base.show
Base.show(io::IO, res::InferenceResult) = println(io, "`InferenceResult` with model", name(res.m))

"""
$(SIGNATURES)

Uses bijectors to make sure to obtain correct parameter values
"""
function construct_result(m::Model, res::RES) where {Model<:AbstractModel, RES}
    params_trained = res.p_trained |> m.mp.st
    return InferenceResult(remake(m,p=params_trained),res) #/!\ new{Model,RES}( is required! 
end

# function construct_result(cm::CM, res::RES) where {CM <: ComposableModel, RES}
#     _ps = res.p_trained
#     params_traineds = [cm.models[i].st(_ps[cm.param_indices[i]]) for i in 1:length(cm.models)]
#     models = [remake(m, p=params_traineds[i]) for (i,m) in enumerate(models)]
#     return InferenceResult(ComposableModel(models...), res) #/!\ new{Model,RES}( is required! 
# end

function loglikelihood(res::InferenceResult, 
                        ode_data::Array, 
                        Σ; 
                        loglike_fn = PiecewiseInference.loglikelihood_lognormal , 
                        u0s = res.res.u0s_trained,
                        p = res.res.p_trained) # we take res.res.p_trained because we would have to transform the parameters otherwise

    θ = [u0s...;p] 
    loss_fn(data, params, pred, rg) = loglike_fn(data, pred, Σ)
    l, _ = piecewise_loss(θ, ode_data, get_kwargs(res.m)[:saveat], model, loss_fn, res.res.ranges; continuity_term=0.)
    return l
end


"""
$(SIGNATURES)

Estimate noise variance, assuming similar noise across all the dimensions of the data.
"""
function estimate_σ(pred::Array, odedata::Array, ::Normal)
    @assert size(pred) == size(odedata)
    RSS = (pred .- odedata).^2
    return sqrt(mean(RSS))
end

function estimate_σ(pred::Array, odedata::Array, ::LogNormal)
    @assert size(pred) == size(odedata)
    logsquare = (log.(pred) .- log.(odedata)).^2
    return sqrt(mean(logsquare[logsquare .< Inf]))
end

function estimate_σ(reseco::InferenceResult, odedata::Array; noisedistrib=LogNormal(), include_ic = true)
    @assert !isempty(reseco.res.pred) "reseco should have been obtained with `save_pred = true`"
    if include_ic
        odedata = hcat([odedata[:,rg] for rg in reseco.res.ranges]...)
        pred = hcat([reseco.res.pred[i] for i in 1:length(reseco.res.ranges)]...)
    else
        odedata = hcat([odedata[:,rg[2:end]] for rg in reseco.res.ranges]...)
        pred = hcat([reseco.res.pred[i][:,2:end] for i in 1:length(reseco.res.ranges)]...)
    end
    return estimate_σ(pred,odedata,noisedistrib)
end


# TODO: test it !
"""
$(SIGNATURES)

"""
function get_var_covar_matrix(reseco::InferenceResult, odedata::Array, σ::Number, loglike_fn = PiecewiseInference.loglikelihood_lognormal)
    likelihood_fn_optim(p) = Econobio.loglikelihood(reseco, odedata, σ; p = p, loglike_fn)
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
compute_cis_normal(reseco::InferenceResult, odedata, p, α, σ) = compute_cis(get_var_covar_matrix(reseco, 
                                                                    odedata,
                                                                    σ, 
                                                                    loglike_fn = PiecewiseInference.loglikelihood_normal), 
                                                                    p, α)

"""
$(SIGNATURES)

"""
compute_cis_lognormal(reseco::InferenceResult,odedata, p, α, σ) = compute_cis(get_var_covar_matrix(reseco, 
                                                                            odedata,
                                                                            σ, 
                                                                            loglike_fn = PiecewiseInference.loglikelihood_lognormal), 
                                                                            p, α)


"""
$(SIGNATURES)

"""
function R2(odedata, pred, ::LogNormal)
    rsstot = log.(odedata) .- mean(log.(odedata), dims=1)
    rssreg = log.(pred) .- log.(odedata)

    padding = (rsstot .< Inf) .* (rssreg .< Inf)

    vartot = sum(abs2,rsstot[padding])
    var_reg = sum(abs2, rssreg[padding])
    R2 = 1 - var_reg / vartot
    return R2
end

