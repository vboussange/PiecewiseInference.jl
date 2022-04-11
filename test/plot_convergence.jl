

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