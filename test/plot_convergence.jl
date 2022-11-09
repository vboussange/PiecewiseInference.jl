cd(@__DIR__)

niters = 1000
datasize = 500
dim_prob = 3
losses = exp.(-(1:niters)/100)

pred = randn(dim_prob,datasize) .+ [0.1, 0.2, 0.3]
data_set = randn(dim_prob,datasize) .+ [0.1, 0.2, 0.3]

pred_ens = randn(dim_prob,datasize, 2) .+ [0.1, 0.2, 0.3]
data_set_ens = randn(dim_prob,datasize, 2) .+ [0.1, 0.2, 0.3]

ranges = [1:251, 251:datasize]
tsteps = 1:datasize
p_labs = [L"p_1", L"p_2"]
p_true = [0.1, 0.2]
θs = exp.(-(1:niters)/10) .+ 1.
p_trained = [0.05, 0.25]


@testset "2 plot only" begin
    fig1 = plot_convergence(losses, pred, data_set, ranges, tsteps;)
    @test isa(fig1, Figure)
end

@testset "3 plots" begin
    fig1 = plot_convergence(losses, 
                            pred, 
                            data_set, 
                            ranges, 
                            tsteps; 
                            p_true = p_true, 
                            p_labs = p_labs,
                            θs = θs, 
                            p_trained = p_trained)
    @test isa(fig1, Figure)
end

@testset "4 plots, EnsembleProblem" begin
    fig1 = plot_convergence(losses, 
                            pred_ens, 
                            data_set_ens, 
                            ranges, 
                            tsteps; 
                            p_true = p_true, 
                            p_labs = p_labs,
                            θs = θs, 
                            p_trained = p_trained)
    @test isa(fig1, Figure)
end

@testset "piecewise MLE" begin

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
    epochs = [2000]
    optimizers = [Adam(0.01)]

    isdir("figures") ? nothing : mkdir("figures") 
    res = piecewise_MLE(p_init = p_init, 
                        group_size = 101, 
                        data_set = ode_data, 
                        prob = prob, 
                        tsteps = tsteps, 
                        alg = Tsit5(), 
                        sensealg =  ForwardDiffSensitivity(),
                        epochs = epochs, 
                        optimizers = optimizers,
                        p_true = p_true,
                        plotting = true,
                        saving_plots = true,
                        saving_dir = "figures/plotting_convergence",
                        info_per_its=100,
                        )
    @test isa(res, InferenceResult)
    isdir("figures") ? rm("figures", recursive=true) : nothing
end