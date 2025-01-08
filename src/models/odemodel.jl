
abstract type AbstractODEModel <: AbstractModel end

"""
$(SIGNATURES)

Returns the `ODEProblem` associated with to `m`.
"""
function get_prob(m::AbstractODEModel, u0, tspan, p)
    prob = ODEProblem(m, u0, tspan, p)
    return prob
end

"""
$(SIGNATURES)

Simulate model `m` and returns an `ODESolution`.  
When provided, keyword arguments overwrite default solving options 
in m.
"""
function simulate(m::AbstractODEModel; u0 = nothing, tspan=nothing, p = nothing, alg = nothing, kwargs...)
    isnothing(u0) ? u0 = get_u0(m) : nothing
    isnothing(tspan) ? tspan = get_tspan(m) : nothing

    # TODO: for now, we assume that `p` contains all model parameters if provided
    # we may want to have `p = merge(p0, p)`, but `merge` only works for non-nested component arrays
    # use cases are e.g. when wants to only train a subset of the model parameters
    # but non trainable parameters could be stored using the model structure
    if isnothing(p) 
        p = get_p(m) 
    end
    # else
    #     # p can be a sub tuple of the full parameter tuple
    #     p0 = get_p(m)
    #     p = merge(p0, p)
    # end
    isnothing(alg) ? alg = get_alg(m) : nothing
    prob = get_prob(m, u0, tspan, p)
    # kwargs erases get_kwargs(m)
    sol = solve(prob, alg; get_kwargs(m)..., kwargs...)
    return sol
end


"""
$SIGNATURES

Generates the skeleton of the model, a `struct` containing details of the numerical implementation.
"""
macro ODEModel(name) 
    expr = quote
        struct $name{MP<:ModelParams} <: AbstractModel
            mp::MP
        end

        $(esc(name))(;mp) = $(esc(name))(mp)
    end
    return expr
end