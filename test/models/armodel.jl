using PiecewiseInference
using UnPack
using ComponentArrays
using OptimizationOptimisers
using Distributions
using Bijectors
using Test
using Random

@ARModel LogisticMap

# model analytic formula
function (m::LogisticMap)(u, p)
    T = eltype(u)
    return p.r .* u .* (one(T) .- u)
end

tspan = (0, 100)
tsteps = 0:1:100


@testset "testing `AnalyticModel`" begin
    p = ComponentArray(r = [3.2])
    u0 = [0.5]
    model = LogisticMap(ModelParams(;p,
                                    u0,
                                    saveat = tsteps
                                    ))
    sol = simulate(model; u0, p)
    # @test isapprox(sol[end], 1/p.b[], rtol=1e-3)
end


p_true =  ComponentArray(r = [3.2])
p_init =  ComponentArray(r = [2])
u0 = [0.5]

model = LogisticMap(ModelParams(;p=p_true,
                                    u0,
                                    saveat = tsteps
                                    ))

sol_data = simulate(model)
ode_data = Array(sol_data)

p_bij = (r = bijector(Uniform(1e-3, 5e0)),)
u0_bij = bijector(Uniform(1e-3,5.))

infprob = InferenceProblem(model, p_init; p_bij, u0_bij)
optimizers = [ADAM(0.01)]
epochs = [1000]
group_nb = 2
batchsizes = [group_nb]
@testset "piecewise inference with `AnalyticModel" begin
    res = inference(infprob;
                        group_nb = group_nb, 
                        data = ode_data, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        batchsizes = batchsizes,
                        multi_threading=false
                        )
    p_trained = get_p_trained(res)
    @test all(isapprox.(p_trained[:r], p_true[:r], atol = 1e-3))
    @test length(res.losses) == sum(epochs) + 1
    @test all(isapprox.(res.u0s_trained[1], u0, atol = 1e-3))
end


@ARModel NNMap 
using Lux
const HlSize = 5
neural_net = Lux.Chain(Lux.Dense(1,HlSize,relu), 
                        Lux.Dense(HlSize,HlSize, relu), 
                        Lux.Dense(HlSize,HlSize, relu), 
                        Lux.Dense(HlSize, 1))

rng = Random.default_rng()
p_nn_init, st = Lux.setup(rng, neural_net)

function (m::NNMap)(u, p)
    y, _ = Lux.apply(neural_net, u, p.p_nn, st)
    return y
end

p_init = ComponentArray(;p_nn = p_nn_init)

model = NNMap(ModelParams(;p=p_init,
                                u0,
                                saveat = tsteps
                                ))

infprob = InferenceProblem(model, p_init)
optimizers = [ADAM(0.01)]
epochs = [1000]
group_nb = 20
batchsizes = [group_nb]
@testset "piecewise inference with `AnalyticModel" begin
    res = inference(infprob;
                        group_nb = group_nb, 
                        data = ode_data, 
                        tsteps = tsteps, 
                        epochs = epochs, 
                        optimizers = optimizers,
                        multi_threading=false
                        )
    p_trained = get_p_trained(res)
    @test length(res.losses) == sum(epochs) + 1

    if false
        p = Plots.plot(tsteps, ode_data[:])
        for (i, r) in enumerate(res.ranges)
            Plots.plot!(tsteps[r], res.pred[i][:],  color="red", linestyle=:dash)
        end
        display(p)
    end
    @test res.losses[end] < 1e-3
end