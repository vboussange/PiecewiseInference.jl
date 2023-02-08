using LinearAlgebra, ParametricModels, OrdinaryDiffEq, SciMLSensitivity
using UnPack
using OptimizationOptimisers
using Test
using PiecewiseInference
using Distributions
using Bijectors

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
# u0_bij = bijector(Uniform(0.,5.))
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

infprob = InferenceProblem(model, p_init; p_bij, u0_bij)


@testset "`InferenceResult``" begin
    res = InferenceResult(infprob, 
                        Inf,
                        p_init, 
                        [u0],
                        [ode_data], 
                        [1:length(tsteps)],
                        [ode_data])
    p_res = get_p_trained(res)
    @test all(p_res[:b] .== p_init[:b])
end

@testset "`InferenceResult`` from `piecewise_ML`" begin
    σ = 0.1

    ode_data_wn = Array(sol_data) 
    ode_data_wn .+=  randn(size(ode_data)) .* σ

    res = piecewise_MLE(infprob;
                        group_size = 101, 
                        data = ode_data_wn, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    u0s_init = res.u0s_trained[1]
    @test length(u0s_init) == length(u0)
    @test any(get_p_trained(res)[:b] .!== p_true[:b])
end

@testset "u0s for `InferenceResult`` from `piecewise_ML_indep_TS`" begin
    tsteps_arr = [tsteps[1:30],tsteps[31:60],tsteps[61:90]] # 3 ≠ time steps with ≠ length

    u0s = [rand(2) .+ 1, rand(2) .+ 1, rand(2) .+ 1]
    ode_datas = []
    for (i,u0) in enumerate(u0s) # generating independent time series
        sol_data = simulate(model; u0, saveat = tsteps_arr[i], sensealg = ForwardDiffSensitivity())
        ode_data = Array(sol_data) 
        ode_data .+=  randn(size(ode_data)) .* 0.1
        push!(ode_datas, ode_data)
    end

    res = piecewise_ML_indep_TS(infprob;
                        data = ode_datas, 
                        group_size = 31, 
                        tsteps = tsteps_arr, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    u0s_init = res.u0s_trained[1][1]
    @test length(u0s_init) == length(u0)

    #TODO: to implement
    # @test (AIC(res, ode_datas, diagm(ones(length(p_init)))) isa Number)
end
