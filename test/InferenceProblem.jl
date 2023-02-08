using LinearAlgebra, ParametricModels, OrdinaryDiffEq, SciMLSensitivity
using Bijectors: Exp, inverse, Identity, Stacked, bijector
using UnPack
using OptimizationOptimisers
using Test
using PiecewiseInference
using Distributions

@model MyModel
function (m::MyModel)(du, u, p, t)
    @unpack b = p
    du .=  0.1 .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:0.5:100.5
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
optimizers = [Adam(0.001)]
epochs = [10]

import PiecewiseInference: get_p_bijector, get_re, get_p
@testset "`InferenceProblem`: parameter transformation in optim space" begin
    infprob = InferenceProblem(model, p_init; p_bij, u0_bij)
    p0_optimspace = PiecewiseInference.get_p(infprob)
    p_trained = inverse(get_p_bijector(infprob))(p0_optimspace)
    p_init_2 = get_re(infprob)(p_trained)
    @test all(p_init_2[:b] .== p_init[:b])
end

@testset "`InferenceProblem`: default behavior" begin
    infprob = InferenceProblem(model, p_init)
    p0_optimspace = PiecewiseInference.get_p(infprob)
    p_trained = inverse(get_p_bijector(infprob))(p0_optimspace)
    p_init_2 = get_re(infprob)(p_trained)
    @test all(p_init_2[:b] .== p_init[:b])
end