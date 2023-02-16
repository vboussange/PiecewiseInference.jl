using LinearAlgebra, ParametricModels, OrdinaryDiffEq, SciMLSensitivity
using UnPack
using OptimizationOptimisers, OptimizationFlux, OptimizationOptimJL
using Test
using PiecewiseInference
using Bijectors
import PiecewiseInference:loss_param_prior
using Optimization
using ComponentArrays

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
p_bij = (bijector(Uniform(1e-3, 5e0)),)
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
@testset "piecewise MLE" begin
    res = piecewise_MLE(infprob;
                        group_nb = group_nb, 
                        data = ode_data, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        batchsizes = batchsizes,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-3))
    @test length(res.losses) == sum(epochs) + 1
    @test all(isapprox.(res.u0s_trained[1], u0, atol = 1e-3))
end

batchsizes = [1]
@testset "piecewise MLE, SGD" begin
    res = piecewise_MLE(infprob;
                        group_nb = group_nb, 
                        data = ode_data, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        batchsizes = batchsizes,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-3 ))
    @test length(res.losses) == sum(epochs) + 1
end

group_nb = 3
batchsizes = [2]
@testset "piecewise MLE, minibatch" begin
    res = piecewise_MLE(infprob; 
                        group_nb = group_nb, 
                        data = ode_data, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        batchsizes = batchsizes,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-3))
    @test length(res.losses) == sum(epochs) + 1
end

@testset "MLE 1 group" begin
    ode_data_wnoise = ode_data .+ randn(size(ode_data)) .* 0.1
    res = piecewise_MLE(infprob;
                        group_nb = 1, 
                        data = ode_data_wnoise, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-1))
    @test length(res.losses) == sum(epochs) + 1
end

@testset "MLE 1 group, LBFGS" begin
    optimizers = [LBFGS()]
    epochs = [5000]
    ode_data_wnoise = ode_data .+ randn(size(ode_data)) .* 0.1
    res = piecewise_MLE(infprob; 
                        group_nb = 1, 
                        data = ode_data_wnoise, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-1))
    @test length(res.losses) <= sum(epochs) + 1
end

@testset "MLE 1 group, ADAM, then LBFGS" begin
    optimizers = [ADAM(0.01), LBFGS()]
    epochs = [1000,200]
    batchsizes = [1,2]
    ode_data_wnoise = ode_data .+ randn(size(ode_data)) .* 0.1
    res = piecewise_MLE(infprob;
                        group_nb = 2, 
                        data = ode_data_wnoise, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-1))
    @test length(res.losses) <= sum(epochs) + 1
end

@testset "piecewise MLE independent TS" begin
    tsteps_arr = [tsteps[1:30],tsteps[31:60],tsteps[61:90]] # 3 ≠ time steps with ≠ length

    u0s = [rand(2) .+ 1, rand(2) .+ 1, rand(2) .+ 1]
    ode_datas = []
    for (i,u0) in enumerate(u0s) # generating independent time series
        sol_data = simulate(model; u0, saveat = tsteps_arr[i], sensealg = ForwardDiffSensitivity())
        ode_data = Array(sol_data) 
        ode_data .+=  randn(size(ode_data)) .* 0.1
        push!(ode_datas, ode_data)
    end

    optimizers = [ADAM(0.001)]
    epochs = [5000]

    res = piecewise_ML_indep_TS(infprob,
                        data = ode_datas, 
                        group_size = 31, 
                        tsteps = tsteps_arr, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-1))
    @test length(res.losses) == sum(epochs) + 1
end

@testset "Initialisation iterative piecewise ML" begin
    group_size_init = 11
    datasize = 100
    ranges_init = get_ranges(;datasize, group_size = group_size_init)
    group_size_2 = 21
    ranges_2 = get_ranges(;datasize, group_size = group_size_2)
    pred_init = [cumsum(ones(3, length(rng)), dims=2) for rng in ranges_init]

    u0_2 = PiecewiseInference._initialise_u0s_iterative_piecewise_ML(pred_init, ranges_init, ranges_2)
    @test u0_2 isa Vector
    @test all([all(u0_2_i .== 1.) for u0_2_i in u0_2])
end

@testset "Iterative piecewise MLE" begin
    ode_data_wnoise = ode_data .+ randn(size(ode_data)) .* 0.1
    group_size_init = 51

    # defining group size and learning rates
    datasize = length(tsteps)
    div_data = divisors(datasize)
    group_sizes = vcat(group_size_init, div_data[div_data .> group_size_init] .+ 1)
    optimizers_array = [[ADAM(0.001)] for _ in 1:length(group_sizes)]
    epochs = [5000]
    res_array = iterative_piecewise_MLE(infprob,
                                        group_sizes = group_sizes, 
                                        optimizers_array = optimizers_array,
                                        epochs = epochs,
                                        data = ode_data_wnoise, 
                                        tsteps = tsteps,)
    p_trained = get_p_trained(res_array[end])
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-1))
    @test length(res_array[end].losses) == sum(epochs) + 1
end


@testset "piecewise MLE, SGD, with priors" begin
    batchsizes = [3]
    group_nb = 3
    param_distrib = Dict(:b => MvNormal([0.23, 0.4], [2., 2.]))
    loss_param_prior(p) = loss_param_prior_from_dict(p, param_distrib)

    infprob = InferenceProblem(model, p_init; p_bij, u0_bij, loss_param_prior)

    res = piecewise_MLE(infprob;
                        group_nb = group_nb, 
                        data = ode_data, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        batchsizes = batchsizes,
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-3 ))
    @test length(res.losses) == sum(epochs) + 1
end

@testset "piecewise MLE, SGD, with priors, adtype = AutoZygote()" begin
    p_true = ComponentArray(b = [0.23, 0.5],)
    p_init= ComponentArray(b = [1., 2.],)
    
    u0 = ones(2)
    p_bij = (bijector(Uniform(1e-3, 5e0)),)
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

    param_distrib = Dict(:b => MvNormal([0.23, 0.4], [2., 2.]))
    loss_param_prior(p) = loss_param_prior_from_dict(p, param_distrib)

    infprob = InferenceProblem(model, p_init; p_bij, u0_bij, loss_param_prior)

    res = piecewise_MLE(infprob;
                        group_nb = group_nb, 
                        data = ode_data, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        batchsizes = batchsizes,
                        adtype = Optimization.AutoZygote()
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-3 ))
    @test length(res.losses) == sum(epochs) + 1
end

# testing stacked bijectors with componentarrays
p_true = ComponentArray(b = [0.23, 0.5],)
lp_1 = [0; [length(p_true[k]) for k in keys(p_true)]...]
p_true = (b = [0.23, 0.5],)
lp_2 = [0;length.(values(p_true))...]
lp_2 == lp_1