
function dudt(du, u, p, t)
    du .=  0.1 .* u .* ( 1. .- p .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])

p_true = [0.23, 0.5]
p_init= [1., 2.]

u0 = ones(2)
prob = ODEProblem(dudt, u0, tspan, p_true)
sol_data = solve(prob, Tsit5(), tspan = tspan, saveat = tsteps, sensealg = ForwardDiffSensitivity())
ode_data = Array(sol_data)
maxiters = Dict("ADAM" => 2000, "BFGS" => 100)

@testset "minibatch MLE" begin
    res = minibatch_MLE(p_init = p_init, 
                        group_size = 101, 
                        data_set = ode_data, 
                        prob = prob, 
                        tsteps = tsteps, 
                        alg = Tsit5(), 
                        sensealg =  ForwardDiffSensitivity(),
                        maxiters = maxiters, 
                        λ = Dict("ADAM" => 0.01, "BFGS" => 0.001),
                        )
    @test all(isapprox.(res.p_trained, p_true, atol = 1e-4 ))
end


@testset "MLE 1 group" begin
    ode_data_wnoise = ode_data .+ randn(size(ode_data)) .* 0.1
    res = minibatch_MLE(p_init = p_init, 
                                group_size = size(ode_data,2) + 1, 
                                data_set = ode_data_wnoise, 
                                prob = prob, 
                                tsteps = tsteps, 
                                alg = Tsit5(), 
                                sensealg = ForwardDiffSensitivity(),
                                maxiters = maxiters)
    @test all( isapprox.(res.p_trained, p_true, rtol = 1e-1))
end

@testset "Iterative minibatch MLE" begin
    ode_data_wnoise = ode_data .+ randn(size(ode_data)) .* 0.1
    group_size_init = 51

    # defining group size and learning rates
    datasize = length(tsteps)
    div_data = divisors(datasize)
    group_sizes = vcat(group_size_init, div_data[div_data .> group_size_init] .+ 1)
    learning_rates = [Dict("ADAM" => 1e-2, "BFGS" => 1e-3) for i in 1:length(group_sizes)]

    res = iterative_minibatch_MLE(group_sizes = group_sizes, 
                                                learning_rates = learning_rates,
                                                p_init = p_init,  
                                                data_set = ode_data_wnoise, 
                                                prob = prob, 
                                                tsteps = tsteps, 
                                                alg = Tsit5(), 
                                                sensealg =  ForwardDiffSensitivity(),
                                                maxiters = maxiters)
    @test all( isapprox.(res.p_trained, p_true, rtol = 1e-1 ))
end

@testset "Initialisation iterative minibatch ML" begin
    group_size_init = 11
    datasize = 100
    ranges_init = group_ranges(datasize, group_size_init)
    group_size_2 = 21
    ranges_2 = group_ranges(datasize, group_size_2)
    pred_init = [cumsum(ones(3, length(rng)), dims=2) for rng in ranges_init]

    u0_2 = EcologyInformedML._initialise_u0s_iterative_minibatch_ML(pred_init, ranges_init, ranges_2)
    @test all(u0_2 .== 1.)
end