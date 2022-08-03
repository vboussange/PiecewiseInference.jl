#=
This file aims at implementing a minibatch loss where the model is something like
model(u0,t,p).

It can encompass models with closed form formula, or ODE models wrapped in an
`AbstractModel`.

This is future work, but should probably be the best model format of MiniBatchInference.jl.

A macro could be implemented to define more simply an AbstractModel by the user.
=#


log_growth2(t, r, b, u0) = (u0 * exp(r * tsteps)) / (1 + b * u0 * (exp(r*t) - 1e0))


"""
$(SIGNATURES)

Minibatch_loss for simple `model(t, p, u0).`
"""
function minibatch_loss(θ, ode_data::Vector, model::AbstractModel, tsteps, loss_function, ranges)
    dim_prob = 1
    nb_group = length(ranges)
    params = _get_param(θ, nb_group, dim_prob) # params of the problem
    u0s = _get_u0s(θ, nb_group, dim_prob)
    l = 0. #loss
    for (i,rg) in ranges
        for t in tsteps[rg]
            pred = model(t, params, u0s[i])
            l += loss_function(pred, ode_data[rg])
        end
    end
    return l
end


struct ModelLog <: AbstractModel
    mp::ModelParams
end
# dists must be a tuple of distributions from Bijectors.jl
function ModelAnalyticalLog(mp, dists)
    @assert length(dists) == 2
    ModelLog(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N])))
end

function (m::ModelAnalyticalLog)(t, p, u0)
    r = getr(p, m)
    b = getb(p, m)
    return (u0 * exp(r * tsteps)) / (1 + b * u0 * (exp(r*t) - 1e0))
end