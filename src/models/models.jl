import Base
function Base.merge(ca::ComponentArray{T}, ca2::ComponentArray{T2}) where {T, T2}
    ax = getaxes(ca)
    ax2 = getaxes(ca2)
    vks = valkeys(ax[1])
    vks2 = valkeys(ax2[1])
    _p = Vector{T}()
    for vk in vks
        @assert !(getproperty(ca2, vk) isa ComponentVector) "Only non-nested `ComponentArray`s are supported by `merge`."
        if vk in vks2
            _vec = vec(getproperty(ca2, vk)) # ca2[vk]
            _p = vcat(_p, _vec)
        else
            _vec = vec(getproperty(ca, vk)) # ca1[vk]
            _p = vcat(_p, _vec)
        end
    end
    # for vk in vks2
    #     if vk not in vks
    #         _vec = vec(getproperty(ca2, vk)) # ca1[vk]
    #         _p = vcat(_p, _vec)
    #         ax = merge(ax, )
    #     end
    # end
    ComponentArray(_p, ax)
end

Base.merge(::Nothing, ca2::ComponentArray{T2}) where {T2} = ca2


# This piece is inspired from https://github.com/jonniedie/ComponentArrays.jl/pull/217
# import ComponentArrays: promote_type, getval, Val, indexmap
# @generated function valkeys(ax::AbstractAxis)
#     idxmap = indexmap(ax)
#     k = Val.(keys(idxmap))
#     return :($k)
# end
# valkeys(ca::ComponentVector) = valkeys(getaxes(ca)[1])

# function merge(cvec1::ComponentVector{T1}, cvec2::ComponentVector{T2}) where {T1, T2}
#     typed_dict = ComponentVector{promote_type(T1, T2)}(cvec1)
#     for key in valkeys(cvec2)
#         keyname = getval(key)
#         val = cvec2[key]
#         typed_dict = eval(:( ComponentArray($typed_dict, $keyname = $val) ))
#     end
#     typed_dict
# end

abstract type AbstractModel end
name(m::AbstractModel) = string(nameof(typeof(m)))
Base.show(io::IO, cm::AbstractModel) = println(io, "`Model` ", name(cm))

struct ModelParams{P,T,U0,A,K}
    p::P # model parameters; we require dictionary or named tuples or componentarrays
    tspan::T # time span
    u0::U0 # initial conditions
    alg::A # alg for ODEsolve
    kwargs::K # kwargs given to solve fn, e.g., saveat
end

import SciMLBase.remake
function remake(mp::ModelParams; 
                p = mp.p, 
                tspan = mp.tspan, 
                u0 = mp.u0, 
                alg = mp.alg, 
                kwargs = mp.kwargs) 
    ModelParams(p, tspan, u0, alg, kwargs)
end
    
# # for the remake fn
# function ModelParams(;p, 
#                     p_bij::PST, 
#                     re, 
#                     tspan, 
#                     u0, 
#                     u0_bij, 
#                     alg, 
#                     dims, 
#                     plength,
#                     kwargs) where PST <: Bijector
#     ModelParams(p, 
#                 p_bij, 
#                 re, 
#                 tspan, 
#                 u0, 
#                 u0_bij, 
#                 alg, 
#                 dims, 
#                 plength,
#                 kwargs)
# end

# model parameters
"""
$(SIGNATURES)

Structure containing the details for the numerical simulation of a model.

# Arguments
- `tspan`: time span of the simulation
- `u0`: initial condition of the simulation
- `alg`: numerical solver
- `kwargs`: extra keyword args provided to the `solve` function.

# Optional
- `p`: default parameter values
# Example
mp = ModelParams()
"""
function ModelParams(; p = nothing, tspan = nothing, u0 = nothing, alg = nothing, kwargs...)
    ModelParams(p,
                tspan,
                u0,
                alg,
                kwargs)
end
ModelParams(p, tspan, u0, alg) = ModelParams(p, tspan, u0, alg, ())

get_mp(m::AbstractModel) = m.mp
get_p(m::AbstractModel) = m.mp.p
get_u0(m::AbstractModel) = m.mp.u0
get_alg(m::AbstractModel) = m.mp.alg
get_tspan(m::AbstractModel) = m.mp.tspan
get_kwargs(m::AbstractModel) = m.mp.kwargs
"""
$SIGNATURES

Returns the dimension of the state variable
"""
get_dims(m::AbstractModel) = length(get_u0(m))


