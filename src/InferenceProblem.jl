Base.@kwdef struct InferenceProblem{M,P,RE,PB,UB}
    m::M
    p0::P
    re::RE
    p_bij::PB
    u0_bij::UB
end

"""
$(SIGNATURES)

## Args
- `model`: a model of type `AbstractModel`.
- `p0`: initial guess of the parameters. Should be a named tuple.
- `p_bij`: a tuple with same length as `p0`, containing bijectors, to constrain parameter values.
- `u0_bij`: a bijector for to constrain state variables `u0`.
"""
function InferenceProblem(model::M, 
                    p0::T,
                    p_bij = fill(Identity{0}(),length(p0)),
                    u0_bij = Identity{0}()) where {M <: AbstractModel, T<: NamedTuple}
    @assert p0 isa NamedTuple
    @assert eltype(p0) <: AbstractArray "The values of `p` must be arrays"
    @assert length(p_bij) == length(values(p0)) "Each element of `p_dist` should correspond to an entry of `p0`"
    lp = [0;length.(values(p0))...]
    idx_st = [sum(lp[1:i])+1:sum(lp[1:i+1]) for i in 1:length(lp)-1]
    p_bij = Stacked(p_bij,idx_st)

    pflat, re = Optimisers.destructure(p0)
    pflat = p_bij(pflat)

    InferenceProblem(model,
                    pflat,
                    re,
                    p_bij,
                    u0_bij)
end

import ParametricModels: get_p, get_mp, get_tspan
get_p(prob::InferenceProblem) = prob.p0
get_p_bijector(prob::InferenceProblem) =prob.p_bij
get_u0_bijector(prob::InferenceProblem) = prob.u0_bij
get_re(prob::InferenceProblem) = prob.re
get_tspan(prob::InferenceProblem) = get_tspan(prob.m)
get_model(prob::InferenceProblem) = prob.m
get_mp(prob::InferenceProblem) = get_mp(get_model(prob))
get_dims(prob::InferenceProblem) = get_dims(get_model(prob))