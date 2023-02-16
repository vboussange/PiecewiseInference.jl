Base.@kwdef struct InferenceProblem{M <: AbstractModel,P,PP,U0P,LL,PB,UB}
    m::M
    p0::P
    loss_param_prior::PP
    loss_u0_prior::U0P
    loss_likelihood::LL
    p_bij::PB
    u0_bij::UB
end

"""
$(SIGNATURES)

## Args
- `model`: a model of type `AbstractModel`.
- `p0`: initial guess of the parameters. Should be a named tuple.

## Optional
- `p_bij`: a dictionary containing bijectors, to constrain parameter values and initial conditions
- `loss_param_prior` is a function with arguments `p::NamedTuple`. Should correspond to parameter priors.
By default, `loss_param_prior(p) = 0`.
- `loss_u0_prior` is a function with arguments `u0_pred, u0_data`. Should correspond to IC priors.
By default it corresponds to RSS between `u0` and the corresponding data point.
- `loss_likelihood` is a function that matches the predictions and the data,
which should have as arguments `data, pred, rng`. By default, it corresponds to the RSS.
"""
function InferenceProblem(model::M, 
                            p0::T;
                            p_bij = nothing,
                            u0_bij = identity,
                            loss_param_prior = _default_param_prior,
                            loss_u0_prior = _default_loss_u0_prior,
                            loss_likelihood = _default_loss_likelihood) where {M <: AbstractModel, T <: ComponentArray}
    # @assert p0 isa NamedTuple
    # @assert eltype(p0) <: AbstractArray "The values of `p` must be arrays"
    @assert length(p_bij) == length(keys(p0)) "Each element of `p_dist` should correspond to an entry of `p0`"
    @assert loss_param_prior(p0) isa Number

    # drawing `u0_pred` and `u0_data` in the optimization space,
    # and projecting it to test `loss_u0_prior`
    u0_pred = randn(get_dims(model)); u0_data = randn(get_dims(model));
    @assert loss_u0_prior(inverse(u0_bij)(u0_pred), inverse(u0_bij)(u0_data)) isa Number
    # TODO: do the same as above for `loss_param_prior` and `loss_likelihood`
    # drawing `data` and `pred` in the optimization space,
    # and projecting it to test `loss_u0_prior`
    # data = randn(get_dims(model), 10); pred = randn(get_dims(model), 10)
    # ...
    # @assert loss_likelihood(data, pred, nothing) isa Number
    if isnothing(p_bij)
        p_bij = Dict{:Symbol,Bijectors.Transform}()
        all_params = vcat(keys(p)...,:u0)
        for k in all_params
            p_bij[k] = identity
        end
    end

    p̃ = to_optim_space(p0, p_bij)

    InferenceProblem(model,
                    p̃,
                    loss_param_prior,
                    loss_u0_prior, 
                    loss_likelihood,
                    p_bij,
                    u0_bij)
end

import ParametricModels: get_p, get_mp, get_tspan
get_p(prob::InferenceProblem) = prob.p0
get_p_bijector(prob::InferenceProblem) =prob.p_bij
get_u0_bijector(prob::InferenceProblem) = prob.u0_bij
get_tspan(prob::InferenceProblem) = get_tspan(prob.m)
get_model(prob::InferenceProblem) = prob.m
get_mp(prob::InferenceProblem) = get_mp(get_model(prob))
import ParametricModels.get_dims
get_dims(prob::InferenceProblem) = get_dims(get_model(prob))
get_loss_likelihood(prob::InferenceProblem) = prob.loss_likelihood
get_loss_param_prior(prob::InferenceProblem) = prob.loss_param_prior
get_loss_u0_prior(prob::InferenceProblem) = prob.loss_u0_prior

#=
Default priors and likelihoods
=#
function _default_loss_likelihood(data, pred, rg)
    l = sum((data - pred).^2)
    return l
end

function _default_loss_u0_prior(u0, u0data)
    l = sum((u0 .- u0data).^2)
    return l
end

function _default_param_prior(p)
    return 0.
end