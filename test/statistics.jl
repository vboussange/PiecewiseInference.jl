using GLM

@testset "likelihood normal" begin
    σ_estim(L, N, M) = exp(-(2 * L / N / M + 1)) / 2 / pi

    N = 2
    p_init = [0.5, 1., 0.2, 0.2]
    tspan = (0.,15.)
    u0_init = [0.1, 0.15]
    tsteps = range(tspan[1], tspan[end], length=1000)
    mymodel = ModelLog(ModelParams(N=N,
                                p = p_init,
                                u0 = u0_init,
                                tspan = tspan,
                                alg = BS3(),
                                kwargs_sol = Dict(:saveat => tsteps,)),
                                (Identity{0}(), Identity{0}()))
    sol = simulate(mymodel)
    true_data = sol |> Array

    σ = 0.8
    odedata = true_data + σ * randn(size(true_data)...)

    # using PyPlot
    # fig = figure()
    # plot(true_data')
    # display(fig)
    # plot(odedata', linestyle = ":")
    # display(fig)

    using LinearAlgebra
    L_mb = MiniBatchInference.loglikelihood_normal(odedata, true_data, σ^2 * LinearAlgebra.I)
    res = ResultEconobio(mymodel,ResultMLE(p_trained = p_init, u0s_trained=[u0_init], ranges = [1:length(tsteps)]))
    L_ec = Econobio.loglikelihood(res, odedata, σ; loglike_fn = MiniBatchInference.loglikelihood_normal)

    @test isapprox(σ_estim(L_mb, size(odedata)...), σ^2, atol = 5e-2)
    @test isapprox(L_mb, L_ec)
end

@testset "likelihood lognormal" begin
    σ_estim(L, N, M, odedata) = exp(-((2 * L + sum(log.(odedata.^2))) / N / M + 1)) / 2 / pi

    N = 2
    p_init = [0.5, 1., 0.2, 0.2]
    tspan = (0.,15.)
    u0_init = [0.1, 0.15]
    tsteps = range(tspan[1], tspan[end], length=1000)
    mymodel = ModelLog(ModelParams(N=N,
                                p = p_init,
                                u0 = u0_init,
                                tspan = tspan,
                                alg = BS3(),
                                kwargs_sol = Dict(:saveat => tsteps,)),
                                (Identity{0}(), Identity{0}()))
    sol = simulate(mymodel)
    true_data = sol |> Array
  

    σ = 0.8
    odedata = true_data .* exp.(σ * randn(size(true_data)...))

    # using PyPlot
    # fig = figure()
    # plot(true_data')
    # display(fig)
    # plot(odedata', linestyle = ":")
    # display(fig)

    # @test pdf(MvLogNormal(zeros(2), sigma^2 * LinearAlgebra.I), 2 * ones(2)) ≈ pdf(LogNormal(0,sigma),2.)^2

    L_mb = MiniBatchInference.loglikelihood_lognormal(odedata, true_data, σ)
    res = ResultEconobio(mymodel,ResultMLE(p_trained = p_init, u0s_trained=[u0_init], ranges = [1:length(tsteps)]))
    L_ec = Econobio.loglikelihood(res, odedata, σ; loglike_fn = MiniBatchInference.loglikelihood_lognormal)

    sigma_estim = σ_estim(L_mb, size(odedata)..., odedata)
    @test isapprox(sigma_estim, σ^2, atol = 5e-2)
    @test isapprox(L_mb, L_ec)
end


@testset "estimate_σ" begin

    N = 2
    p_init = [0.5, 1., 0.2, 0.2]
    tspan = (0.,15.)
    u0_init = [0.1, 0.15]
    tsteps = range(tspan[1], tspan[end], length=1000)
    mymodel = ModelLog(ModelParams(N=N,
                                p = p_init,
                                u0 = u0_init,
                                tspan = tspan,
                                alg = BS3(),
                                kwargs_sol = Dict(:saveat => tsteps,)),
                                (Identity{0}(), Identity{0}()))
    sol = simulate(mymodel)
    true_data = sol |> Array
  

    for σ in 0.1:0.1:0.8
        odedata = true_data .* exp.(σ * randn(size(true_data)...))
        @test isapprox(estimate_σ(true_data,odedata,LogNormal()), σ, rtol = 5e-2)

        odedata = true_data .+ σ * randn(size(true_data)...)
        @test isapprox(estimate_σ(true_data,odedata,Normal()), σ, rtol = 5e-2)
    end

end


# TODO: confidence intervals tests not implemented
@testset "confidence intervals normally distributed noise" begin
    σ_estim(L, N, M) = exp(-(2 * L / N / M + 1)) / 2 / pi

    N = 2
    p_init = [0.5, 1., 0.2, 0.2]
    tspan = (0.,15.)
    u0_init = [0.1, 0.15]
    tsteps = range(tspan[1], tspan[end], length=1000)
    mymodel = ModelLog(ModelParams(N=N,
                                p = p_init,
                                u0 = u0_init,
                                tspan = tspan,
                                alg = BS3(),
                                kwargs_sol = Dict(:saveat => tsteps,)),
                                (Identity{0}(), Identity{0}()))
    sol = simulate(mymodel)
    true_data = sol |> Array

    σ = 1000.0
    odedata = true_data + σ * randn(size(true_data)...)

    res = ResultEconobio(mymodel,ResultMLE(p_trained = p_init, u0s_trained=[u0_init], ranges = [1:length(tsteps)]))
    L_ec(p) = - Econobio.loglikelihood(res, odedata, 1. .* LinearAlgebra.I; loglike_fn = MiniBatchInference.loglikelihood_normal, p = p)
    numerical_hessian = ForwardDiff.hessian(L_ec, p_init)
    var_cov_matrix = inv(numerical_hessian)
    display(var_cov_matrix)
    t_stats = p_init ./ sqrt.(-diag(var_cov_matrix))    
end

@testset "confidence intervals lognormally distributed noise" begin

    N = 2
    p_init = [0.5, 1., 0.2, 0.2]
    tspan = (0.,15.)
    u0_init = [0.1, 0.15]
    tsteps = range(tspan[1], tspan[end], length=1000)
    mymodel = ModelLog(ModelParams(N=N,
                                p = p_init,
                                u0 = u0_init,
                                tspan = tspan,
                                alg = BS3(),
                                kwargs_sol = Dict(:saveat => tsteps,)),
                                (Identity{0}(), Identity{0}()))
    sol = simulate(mymodel)
    true_data = sol |> Array
  

    σ = 100.
    odedata = true_data .* exp.(σ * randn(size(true_data)...))

    # using PyPlot
    # fig = figure()
    # plot(true_data')
    # display(fig)
    # plot(odedata', linestyle = ":")
    # display(fig)

    # @test pdf(MvLogNormal(zeros(2), sigma^2 * LinearAlgebra.I), 2 * ones(2)) ≈ pdf(LogNormal(0,sigma),2.)^2

    res = ResultEconobio(mymodel,ResultMLE(p_trained = p_init, u0s_trained=[u0_init], ranges = [1:length(tsteps)]))
    L_ec(p) = Econobio.loglikelihood(res, odedata, σ; loglike_fn = MiniBatchInference.loglikelihood_lognormal, p = p)
    using ForwardDiff
    numerical_hessian = ForwardDiff.hessian(L_ec, p_init)
    var_cov_matrix = inv(numerical_hessian)
    display(var_cov_matrix)
    t_stats = p_init ./ sqrt.(-diag(var_cov_matrix))        

end

@testset "R2" begin
    x = 1:0.1:10 |> collect
    y = x .* exp.(0.1 *  randn(length(x)))
    df_test = DataFrame("x" => x, "y" => y)
    ols = lm(@formula(log(y) ~ log(x)), df_test)
    r2_lm = r2(ols)
    ŷ = exp.(predict(ols))
    r2_cust = R2(y, ŷ, LogNormal())
    @test isapprox(r2_lm, r2_cust)
end