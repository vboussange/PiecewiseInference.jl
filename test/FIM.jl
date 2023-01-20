#= 
This script aims at testing the equivalence between 
FIM from strouwn and the one from Yazdani.

We test this on the coupled logistic map
=#
using OrdinaryDiffEq
using SciMLSensitivity
using Test

# Create a name for saving ( basically a prefix )
T = 0.25
tspan = (0.0, 6.0)

dim_prob = 3 # number of state variables
function dudt(du, u, p, t)
    r = p[1:dim_prob]
    A = reshape(p[dim_prob+1:end], dim_prob, dim_prob)
    du .= r .* u .* (1.0 .- A * u)
end

# Define the experimental parameters
r = [3.6, 3.0, 3.0]
A = [1. 0.2 0.2; 0.2 1. -0.2; 0.2 -0.2 1.]
p_true = [r; A[:]]
# labels for the parameters
lab = [[latexstring("r_$i") for i in 1:dim_prob]; [latexstring("a_{$i,$j}") for i in 1:dim_prob, j in 1:dim_prob][:];]
u0_true = [0.2, 0.1, 0.3]
Σ = diagm(ones(dim_prob))

##########################
## solving the  problem ##
##########################
prob = ODEProblem(dudt, u0_true, tspan, p_true)
sol = solve(prob, saveat=T, alg = Tsit5())
using Plots
Plots.plot(sol)

#####################################
## solving the sensitivity problem ##
#####################################
@time FIM_yaz = FIM_yazdani(dudt, u0_true, tspan, p_true, Σ)

function predict(θ)
    prob = ODEProblem(dudt, u0_true, tspan, θ)
    sol = solve(prob, saveat=T, alg = Tsit5())
    return sol |> Array
end
@time FIM_str = FIM_strouwen(predict, p_true, Σ)

# remark: FIM_Yaz seems faster by a factor of 10
@test all(isapprox.(FIM_yaz, FIM_str, rtol=1e-3))