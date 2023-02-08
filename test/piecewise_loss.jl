using PiecewiseInference, Test, ForwardDiff, OrdinaryDiffEq
using SciMLSensitivity:ForwardDiffSensitivity
using OptimizationOptimJL:BFGS
using OptimizationOptimisers:Adam
using ParametricModels
using Distributions
using UnPack

@model MyModel
function (m::MyModel)(du, u, p, t)
    @unpack r, b = p
    du .=  r .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])
datasize = length(tsteps)
ranges = [1:101, 101:datasize]

p_true = (r = [0.5, 1.], b = [0.23, 0.5],)
p_init= (r = [0.7, 1.2], b = [0.2, 0.2],)

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

# making sure we have good data
# figure()
# plot(tsteps, sol_data')
# gcf()
θ = [ode_data[:,first.(ranges),:][:];p_init[:r];p_init[:b]]
infprob = InferenceProblem(model, p_init)
 
@testset "Testing correct behavior `piecewise_loss`" begin
    l, pred = piecewise_loss(infprob,
                        θ, 
                        ode_data, 
                        tsteps, 
                        ranges,
                        1:length(ranges))
    @test isa(l, Number)
    @test isa(pred, Vector)
end

@testset "Testing differentiability `piecewise_loss`" begin
    _loss(θ) = piecewise_loss(infprob,
                                θ, 
                                ode_data, 
                                tsteps, 
                                ranges,
                                1:length(ranges))[1]
    l = _loss(θ)
    mygrad = ForwardDiff.gradient(_loss, θ)
    @test length(mygrad) == length(θ)
end