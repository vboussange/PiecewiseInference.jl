using LinearAlgebra, ParametricModels, OrdinaryDiffEq, SciMLSensitivity
using UnPack
using OptimizationOptimisers
using Test
using PiecewiseInference
using Distributions
using Bijectors
using ForwardDiff

σ_noise = 0.1

@model MyModel
function (m::MyModel)(du, u, p, t)
    @unpack b = p
    du .=  0.1 .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:1.:100.5
tspan = (tsteps[1], tsteps[end])

p_true = (b = [0.23, 0.5],)
p_init= (b = [1., 2.],)

p_bij = (bijector(Uniform(0.,3.)),)
u0_bij = bijector(Uniform(0.,5.))

u0 = ones(2)
mp = ModelParams(;p = p_true, 
                tspan,
                u0, 
                alg = BS3(),
                sensealg = ForwardDiffSensitivity(),
                saveat = tsteps, 
                )
model = MyModel(mp)
sol_data = simulate(model)
ode_data = Array(sol_data)
ode_data_w_noise = ode_data .* exp.(σ_noise .* randn(size(ode_data)))

ranges = get_ranges(;datasize = length(tsteps), group_size = 10)

function loss_fn_lognormal_distrib(data, pred, noise_distrib)
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    l = 0.

    # observations
    ϵ = log.(data) .- log.(pred)
    for i in 1:size(ϵ, 2)
        l += logpdf(noise_distrib, ϵ[:, i])
    end
    # l /= size(ϵ, 2) # TODO: this bit is unclear

    if l isa Number # preventing any other reason for Zygote to crash
        return - l
    else 
        return Inf
    end
end


loss_likelihood(data, pred, rng) = loss_fn_lognormal_distrib(data, pred, MvNormal(σ_noise^2 * ones(2)))
infprob = InferenceProblem(model, p_init; p_bij, u0_bij, loss_likelihood)

u0s = [ode_data[:, first(rg)] for rg in ranges]

@test loglikelihood(ode_data_w_noise, tsteps, ranges, infprob, p_true, u0s) isa Number
@test get_evidence(ode_data_w_noise, tsteps, ranges, infprob, p_true, u0s) isa Number