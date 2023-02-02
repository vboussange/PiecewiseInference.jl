"""
$(SIGNATURES)
Get ranges that partition data of length `datasize` in groups of `groupsize` observations.
If the data isn't perfectly dividable by `groupsize`, the last group contains
the reminding observations.
Taken from https://github.com/SciML/DiffEqFlux.jl/blob/80c4247c19860d0422211d6a65283d896eeaa831/src/multiple_shooting.jl#L273-L303
```julia
group_ranges(datasize, groupsize)
```
Arguments:
- `datasize`: amount of data points to be partitioned
- `groupsize`: maximum amount of observations in each group
Example:
```julia-repl
julia> group_ranges(10, 5)
3-element Vector{UnitRange{Int64}}:
 1:5
 5:9
 9:10
```
"""
function group_ranges_gs(datasize::Integer, group_size::Integer)
    2 <= group_size <= datasize || throw(
        DomainError(
            group_size,
            "datasize must be positive and group_size must to be within [2, datasize]",
        ),
    )
    return [i:min(datasize, i + group_size - 1) for i in 1:group_size-1:datasize-1]
end

"""
$(SIGNATURES)

    Similar to `group_ranges`, except that it takes as arguments the nb of groups wanted, `group_nb`
"""
function group_ranges_gn(datasize::Integer, group_nb::Integer)
    group_size = ceil(Integer, datasize/group_nb) + 1
    group_ranges_gs(datasize, group_size)
end

"""
    AIC(RSS, k, m)

Calculate AIC of a model given its `RSS`, 
`k` its number of parameters, 
and `m` the number of observations.
"""
function AIC(RSS, k, m)
    aic = m * log(RSS / m) + 2*k
    return aic
end

"""
    AICc(aic, k, m)

Calculate AIC corrected of a model given its `aic`, 
`k` its number of parameters, 
and `m` the number of observations 
(if d variables and T time steps, m = d * T). 
The formula is taken from Mangan2017.
"""
function AICc(aic, k, m)
    aic_c = aic + 2 * (k + 1) * (k + 2) / (m - k - 2)
    return aic_c
end

"""
AICc_TREE(RSS, k, m)

Calculate aic of a model given its `RSS`, 
`k` its number of parameters, 
and `m` the number of observations.
The formula is taken from TREE article.
"""
function AICc_TREE(RSS, k, m)
    aic = m * log(RSS / m) + 2*k
    aic_c = aic + 2 * (k + 1) * (k + 2) / (m - k - 2)
    return aic_c
end

# calculation of covariates
"""
    moments!(xx, x)
reurns the moments of `x` as a vector stored in `xx`
* Args
- `xx` : the covariate vector
- `x` :  the state variable vector
"""
function moments!(xx, x)
    k = 1
    for i in 1:dim_prob
        for j in i:dim_prob
            xx[k] = x[i] * x[j]
            k += 1
        end
    end
end

"""
    moments(x)
reurns the moments of x as a vector
* Args
- `xx` : the covariate vector
- `x` :  the state variable vector
"""
function moments(x)
    N = length(x)
    xx = similar(x, N * (N + 1) / 2 |> Int)
    k = 1
    for i in 1:N
        for j in i:N
            xx[k] = x[i] * x[j]
            k += 1
        end
    end
    return xx
end

"""
    FIM_strouwen(predict, θ, Σ)
Returns the FIM matrix associated to `predict` and evaluated at `θ`
taken from https://arnostrouwen.com/post/dynamic-experimental-design/
- `predict` is a function that takes `θ` as a parameter and returns an array 
    - with dim=1 corresponding to state variables and 
    - dim = 2 corresponding to the time steps
- `θ` the parameter vector where to evaluate the FIM
- `Σ` is the variance-covariance matrix of the observation errors
"""
function FIM_strouwen(predict, θ, Σ)
    sol = predict(θ)
    n_θ = length(θ)
    n_u, n_t = size(sol)
    FIM = zeros(eltype(sol), n_θ, n_θ)
    jac = ForwardDiff.jacobian(predict, θ)
    for k in 1:n_t
        du_dθ = jac[(k-1)*n_u+1:k*n_u,:]
        FIM += du_dθ'*inv(Σ)*du_dθ
    end
    return FIM
end

"""
    FIM_yazdani(dudt, u0_true, tspan, p_true, Σ)
Returns the FIM matrix associated to problem defined by
`prob = ODEProblem(dudt, u0_true, tspan, p_true)`.
`Σ` is the variance-covariance matrix of the observation errors
"""
function FIM_yazdani(dudt, u0_true, tspan, tsteps, p_true, Σ)
    prob_single = ODELocalSensitivityProblem(dudt, u0_true, tspan, p_true)
    sol_single = solve(prob_single, saveat=tsteps, alg = Tsit5())
    x_single, dp = extract_local_sensitivities(sol_single,)
    Nt = length(dp[1][1,:]) # nb of time steps
    Nstate = length(dp[1][:,1]) # nb of state variables
    Nparam = length(dp[:,1]) # nb of parameters
    FIM = zeros(Float64, Nparam, Nparam) # fisher information matrix
    G = zeros(Float64, Nstate, Nparam) # storing the sensitivities for all time steps
    for i in 1:Nt
        S = reshape(dp[1][:,i], (Nstate,1))
        for j = 2:Nparam # this looks necessary
            S = hcat(S, reshape(dp[j][:,i], (Nstate,1)))
        end
        FIM += S' * inv(Σ) * S
    end
    return FIM
end


"""
    divisors(n)
Returns all divisors of `n`, sorted.
"""
function divisors(n)
    divs = eltype(n)[]
    for i in 1:n
        if n % i == 0
            push!(divs, i)
        end
    end
    return sort!(divs)
end

"""
$SIGNATURES

Prints in a nice format the NamedTuple or Dict `p_trained`.
If `p_true` provided, also display its values for comparisions.
"""
function pretty_print(p_trained, p_true = nothing)
    for k in keys(p_trained)
        println(k, "_trained = ")
        display(p_trained[k])
        println("")
        if !isnothing(p_true)
            println(k,  "true = ")
            display(p_true[k])
        end
        println("*********")
    end
end