
abstract type AbstractAnalyticModel <: AbstractModel end
import ParametricModels:simulate
function simulate(m::AbstractAnalyticModel; tspan = nothing, u0 = nothing, saveat=nothing, p = nothing, kwargs...)
    isnothing(u0) && (u0 = get_u0(m))
    isnothing(tspan) && (tspan = get_tspan(m))
    isnothing(saveat) && (saveat = m.mp.kwargs[:saveat])

    if isnothing(p) 
        p = get_p(m) 
    else
        # p can be a sub tuple of the full parameter tuple
        p0 = get_p(m)
        p = merge(p0, p)
    end

    sim = m.(Ref(u0), Ref(p), saveat)
    
    return hcat(sim...)
end

"""
$SIGNATURES

Generates the skeleton of the model, a `struct` containing details of the numerical implementation.
"""
macro AnalyticModel(name) 
    expr = quote
        struct $name{MP<:ModelParams} <: AbstractAnalyticModel
            mp::MP
        end

        $(esc(name))(;mp) = $(esc(name))(mp)
    end
    return expr
end