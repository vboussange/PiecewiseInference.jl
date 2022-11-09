using ForwardDiff
@model MyModel
function (m::MyModel)(du, u, p, t)
    @unpack r, b = p
    du .=  r .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])
datasize = length(tsteps)
ranges = [1:101, 101:datasize]

p_true = (r = [0.5, 1.], b = [0.23, 0.5],)
p_init= (r = [0.7, 1.2], b = [0.2, 0.2],)

u0 = ones(2)
mp = ModelParams(p_true, 
                tspan,
                u0, 
                BS3(),
                sensealg = ForwardDiffSensitivity();
                saveat = tsteps, 
                )
model = MyModel(mp)
sol_data = simulate(model)
ode_data = Array(sol_data)

loss_function(data, params, pred, rg) = sum(abs2, data - pred)

# making sure we have good data
# figure()
# plot(tsteps, sol_data')
# gcf()
θ = [ode_data[:,first.(ranges),:][:];p_init[:r];p_init[:b]]
 
@testset "Testing correct behavior `piecewise_loss`" begin
    l, pred = piecewise_loss(θ, 
                        ode_data, 
                        tsteps, 
                        model, 
                        loss_function, 
                        ranges,
                        1:length(ranges), # idx_ranges
                        sensealg = ForwardDiffSensitivity())
    @test isa(l, Number)
    @test isa(pred, Vector)
end

@testset "Testing differentiability `piecewise_loss`" begin
    _loss(θ) = piecewise_loss(θ, 
                        ode_data, 
                        tsteps, 
                        model, 
                        loss_function, 
                        ranges, 
                        1:length(ranges), # idx_ranges
                        sensealg = ForwardDiffSensitivity())[1]
    l = _loss(θ)
    mygrad = ForwardDiff.gradient(_loss, θ)
    @test length(mygrad) == length(θ)
end