function dudt(du, u, p, t)
    du .=  0.1 .* u .* ( 1. .- p .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])
datasize = length(tsteps)
ranges = [1:101, 101:datasize]

p_true = [0.23, 0.5]
p_init= [1., 2.]

u0 = ones(2)
prob = ODEProblem(dudt, u0, tspan, p_true)
sol_data = solve(prob, Tsit5(), saveat = tsteps, sensealg = ForwardDiffSensitivity(), kwargshandle=KeywordArgError)
ode_data = Array(sol_data)

loss_function(data, pred) = sum(abs2, data - pred)

# making sure we have good data
# figure()
# plot(tsteps, sol_data')
# gcf()
θ = [ode_data[:,first.(ranges),:][:];p_init]
 
@testset "Testing correct behavior `minibatch_loss`" begin
    l, pred = minibatch_loss(θ, 
                        ode_data, 
                        tsteps, 
                        prob, 
                        loss_function, 
                        Tsit5(), 
                        ranges, 
                        sensealg = ForwardDiffSensitivity())
    @test isa(l, Number)
    @test isa(pred, Vector)
end

@testset "Testing differentiability `minibatch_loss`" begin
    _loss(θ) = minibatch_loss(θ, 
                        ode_data, 
                        tsteps, 
                        prob, 
                        loss_function, 
                        Tsit5(), 
                        ranges, 
                        sensealg = ForwardDiffSensitivity())[1]
    l = _loss(θ)
    mygrad = ForwardDiff.gradient(_loss, θ)
    @test length(mygrad) == length(θ)
end