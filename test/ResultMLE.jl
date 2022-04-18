
function dudt(du, u, p, t)
    du .=  0.1 .* u .* ( 1. .- p .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])

p_true = [0.23, 0.5]
p_init= [1., 2.]

u0 = ones(2)
prob = ODEProblem(dudt, u0, tspan, p_true)
sol_data = solve(prob, Tsit5(), saveat = tsteps, sensealg = ForwardDiffSensitivity())
ode_data = Array(sol_data)
optimizers = [ADAM(0.001)]
maxiters = [10]

@testset "get_u0s for `ResultMLE`` from `minibatch_ML`" begin
    res = minibatch_MLE(p_init = p_init, 
                        group_size = 101, 
                        data_set = ode_data, 
                        prob = prob, 
                        tsteps = tsteps, 
                        alg = Tsit5(), 
                        sensealg =  ForwardDiffSensitivity(),
                        maxiters = maxiters, 
                        optimizers = optimizers,
                        )
    u0s_init = get_u0s(res)[1]
    @test length(u0s_init) == length(u0)
end

@testset "get_u0s for `ResultMLE`` from `minibatch_ML_indep_TS`" begin
    tsteps_arr = [tsteps[1:30],tsteps[31:60],tsteps[61:90]] # 3 ≠ time steps with ≠ length

    u0s = [rand(2) .+ 1, rand(2) .+ 1, rand(2) .+ 1]
    ode_datas = []
    for (i,u0) in enumerate(u0s) # generating independent time series
        prob = ODEProblem(dudt, u0, tspan, p_true)
        sol_data = solve(prob, Tsit5(), saveat = tsteps_arr[i], sensealg = ForwardDiffSensitivity())
        ode_data = Array(sol_data) 
        ode_data .+=  randn(size(ode_data)) .* 0.1
        push!(ode_datas, ode_data)
    end

    res = minibatch_ML_indep_TS(data_set = ode_datas, 
                        group_size = 31, 
                        tsteps = tsteps_arr, 
                        p_init = p_init, 
                        prob = prob, 
                        alg = Tsit5(), 
                        sensealg =  ForwardDiffSensitivity(),
                        maxiters = maxiters, 
                        optimizers = optimizers,
                        )
    u0s_init = get_u0s(res)[1][1]
    @test length(u0s_init) == length(u0)
end
