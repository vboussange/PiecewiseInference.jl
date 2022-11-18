Base.@kwdef struct InferenceProblem{M,P,RE,PB,UB}
    m::M
    p0::P
    re::RE
    p_bij::PB
    u0_bij::UB
end

function InferenceProblem(model::M, 
                    p0,
                    p_bij = fill(Identity{0}(),length(p0)),
                    u0_bij = Identity{0}()) where M <: AbstractModel
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

get_p(prob::InferenceProblem) = prob.p0
get_p_bijector(prob::InferenceProblem) =prob.p_bij
get_u0_bijector(prob::InferenceProblem) = prob.u0_bij
get_re(prob::InferenceProblem) = prob.re
get_tspan(prob::InferenceProblem) = get_tspan(prob.m)
get_model(prob::InferenceProblem) = prob.m
get_mp(prob::InferenceProblem) = get_mp(get_model(prob))

function simulate(prob::InferenceProblem, u0, tspan, p, saveat)
    model = get_model(prob)
    p = inverse(get_p_bijector(prob))(p) # projecting p in true parameter space
    u0 = inverse(get_u0_bijector(prob))(u0) # projecting p in true parameter space
    p_tuple = get_re(prob)(p)
    sol = ParametricModels.simulate(model; p=p_tuple, u0, tspan, saveat)
    return sol
end