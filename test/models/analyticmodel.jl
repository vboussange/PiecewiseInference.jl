using PiecewiseInference
using UnPack
using ComponentArrays
using OptimizationOptimisers
using Distributions
using Bijectors

@AnalyticModel LogisticModel

# model analytic formula
function (m::LogisticModel)(u0, t0, p, t)
    T = eltype(u0)
    @unpack r, b = p

    @. u0 / (exp(-r * (t - t0)) + b * u0 * (one(T) - exp(-r * (t - t0))))
end

tspan = (0., 100.)
tsteps = range(tspan[1], tspan[end], length=100)


@testset "testing `AnalyticModel`" begin
    p = ComponentArray(r = [10.], b = [4.])
    u0 = rand(1)
    model = LogisticModel(ModelParams(;p,
                                    tspan,
                                    u0,
                                    saveat = tsteps
                                    ))
    sol = simulate(model; u0, p)
    @test isapprox(sol[end], 1/p.b[], rtol=1e-3)
end


@testset "Piecewise inference with `AnalyticModel`" begin

    p_true = ComponentArray(r = [0.5], b = [0.23])
    p_init= ComponentArray(r = [0.2], b = [1.])

    u0 = rand(1)
    model = LogisticModel(ModelParams(;p=p_true,
                                        tspan,
                                        u0,
                                        saveat = tsteps
                                        ))

    sol_data = simulate(model)
    ode_data = Array(sol_data)

    p_bij = (b = bijector(Uniform(1e-3, 5e0)), 
            r = bijector(Uniform(1e-3, 5e0)))
    u0_bij = bijector(Uniform(1e-3,5.))

    infprob = InferenceProblem(model, p_init; p_bij, u0_bij)
    optimizers = [ADAM(0.01)]
    epochs = [4000]
    group_nb = 2
    batchsizes = [group_nb]

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
    @test all(isapprox.(p_trained[:b], p_true[:b], atol = 1e-3))
    @test length(res.losses) == sum(epochs) 
    @test all(isapprox.(res.u0s_trained[1], u0, atol = 1e-3))
end
