using LinearAlgebra, ParametricModels, OrdinaryDiffEq, DiffEqSensitivity
using Bijectors: Exp, inverse, Identity, Stacked
using UnPack
using OptimizationOptimisers
using Test
using MiniBatchInference

@model MyModel
function (m::MyModel)(du, u, p, t)
    @unpack b = p
    du .=  0.1 .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])

p_true = (b = [0.23, 0.5],)
p_init= (b = [1., 2.],)

u0 = ones(2)
mp = ModelParams(p_true, 
                tspan,
                u0, 
                BS3(),
                sensealg = ForwardDiffSensitivity();
                saveat = tsteps, 
                )
model = MyModel(mp)
sol_data = simulate(model)
ode_data = Array(sol_data)
optimizers = [Adam(0.001)]
epochs = [10]

@testset "`ResultMLE`` from `minibatch_ML`" begin
    res = minibatch_MLE(p_init = p_init, 
                        group_size = 101, 
                        data_set = ode_data, 
                        model = model, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    u0s_init = res.u0s_trained[1]
    @test length(u0s_init) == length(u0)

    @test (AIC(res, ode_data, diagm(ones(length(u0)))) isa Number)
end

@testset "u0s for `ResultMLE`` from `minibatch_ML_indep_TS`" begin
    tsteps_arr = [tsteps[1:30],tsteps[31:60],tsteps[61:90]] # 3 ≠ time steps with ≠ length

    u0s = [rand(2) .+ 1, rand(2) .+ 1, rand(2) .+ 1]
    ode_datas = []
    for (i,u0) in enumerate(u0s) # generating independent time series
        sol_data = simulate(model; u0, saveat = tsteps_arr[i], sensealg = ForwardDiffSensitivity())
        ode_data = Array(sol_data) 
        ode_data .+=  randn(size(ode_data)) .* 0.1
        push!(ode_datas, ode_data)
    end

    res = minibatch_ML_indep_TS(data_set = ode_datas, 
                        group_size = 31, 
                        tsteps = tsteps_arr, 
                        p_init = p_init, 
                        model = model, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    u0s_init = res.u0s_trained[1][1]
    @test length(u0s_init) == length(u0)

    #TODO: to implement
    # @test (AIC(res, ode_datas, diagm(ones(length(p_init)))) isa Number)
end
