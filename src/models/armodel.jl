
abstract type AbstractARModel <: AbstractModel end

function simulate(m::AbstractARModel; u0 = nothing, saveat=nothing, p = nothing, kwargs...)
    isnothing(u0) && (u0 = get_u0(m))
    isnothing(saveat) && (saveat = m.mp.kwargs[:saveat])

    if isnothing(p) 
        p = get_p(m) 
    else
        # p can be a sub tuple of the full parameter tuple
        # p = get_p(m)
        # p = merge(p0, p)
    end

    T = eltype(u0)
    u = Array{T}(undef, length(u0), length(saveat))
    u[:,1] .= u0
    for i in 2:length(saveat)
        u[:,i] .= m(u[:, i-1], p)
    end
    return u
end


"""
$SIGNATURES

Generates the skeleton of the model, a `struct` containing details of the numerical implementation.
"""
macro ARModel(name) 
    expr = quote
        struct $name{MP<:ModelParams} <: AbstractARModel
            mp::MP
        end

        $(esc(name))(;mp) = $(esc(name))(mp)
    end
    return expr
end