
abstract type AbstractARModel <: AbstractModel end

"""
$SIGNATURES

- TODO: as of current implementation, all parameters must be contained in `p` if `p` is provided. This comes from a bug with the `merge` function. It must be fixed. 
"""
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

Creates an auto-regressive model, with signature `model(p, t)`.
Under the hood, generate a `struct` containing details of the numerical implementation.

# Example

```julia
@ARModel LogisticMap


# model definition
function (m::LogisticMap)(u, p)
    T = eltype(u)
    return p.r .* u .* (one(T) .- u)
end

tspan = (0, 100)
tsteps = 0:1:100

p = ComponentArray(r = [3.2])
u0 = [0.5]
model = LogisticMap(ModelParams(;p,
                                u0,
                                saveat = tsteps
                                ))
sol = simulate(model; u0, p)
```
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