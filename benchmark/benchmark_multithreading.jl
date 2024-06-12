#=
Benchmarking threads vs non threads
=#
using PiecewiseInference
using SciMLSensitivity
using OrdinaryDiffEq
using ComponentArrays
using Bijectors
using UnPack
using OptimizationOptimisers, OptimizationFlux, OptimizationOptimJL

@model MyModel
function (m::MyModel)(du, u, p, t)
    @unpack b = p
    du .=  0.1 .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])

p_true = ComponentArray(b = [0.23, 0.5],)
p_init= ComponentArray(b = [1., 2.],)

u0 = ones(2)
p_bij = (b = bijector(Uniform(1e-3, 5e0)),)
u0_bij = bijector(Uniform(1e-3,5.))

mp = ModelParams(; p = p_true, 
                tspan,
                u0, 
                alg = BS3(),
                sensealg = ForwardDiffSensitivity(),
                saveat = tsteps, 
                )
model = MyModel(mp)
sol_data = simulate(model)
ode_data = Array(sol_data)

infprob = InferenceProblem(model, p_init; p_bij, u0_bij)
optimizers = [ADAM(0.001)]
epochs = [4000]
group_nb = 2
batchsizes = [group_nb]

# Multithreading
@time res = inference(infprob;
                    group_nb = group_nb, 
                    data = ode_data, 
                    tsteps = tsteps, 
                    epochs = epochs, 
                    optimizers = optimizers,
                    batchsizes = batchsizes,
                    multi_threading = true
                    )
# 0.768009 seconds (5.94 M allocations: 914.722 MiB, 11.30% gc time)

# No Multithreading
@time res = inference(infprob;
                    group_nb = group_nb, 
                    data = ode_data, 
                    tsteps = tsteps, 
                    epochs = epochs, 
                    optimizers = optimizers,
                    batchsizes = batchsizes,
                    multi_threading = false
                    )
# 0.812567 seconds (5.73 M allocations: 879.842 MiB, 11.18% gc time)