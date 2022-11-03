""" 
$(SIGNATURES)

Container for ouputs of MLE.

# Notes
`res = ResultMLE()` has all fields empty but `res.minloss` which is set to `Inf`.

"""
Base.@kwdef struct ResultMLE{M,P,U0,Pr,R,L}
    minloss::M = Inf
    p_trained::P = []
    u0s_trained::U0 = [] #  initial condition vector estimated`[u_0_1, ..., u_0_n]`
                 # In the case of independent time series, 
                 # `[[u_0_TS1_1, ..., u_0_TS1_n],...,[u_0_TS1_1,...]]`, matching the format of `res.pred`.
    pred::Pr = []
    ranges::R = []
    losses::L = []
end


"""
$(SIGNATURES)

"""
Base.@kwdef struct InferenceResult{Model<:AbstractModel, RES}
    m::Model
    res::RES
end
import Base.show
Base.show(io::IO, res::InferenceResult) = println(io, "`InferenceResult` with model", name(res.m))

"""
$(SIGNATURES)

Uses bijectors to make sure to obtain correct parameter values
"""
function construct_result(m::Model, res::RES) where {Model<:AbstractModel, RES}
    params_trained = res.p_trained |> m.mp.st
    return InferenceResult(remake(m,p=params_trained),res) #/!\ new{Model,RES}( is required! 
end

get_p_trained(res::InferenceResult) = res.m.p

# function construct_result(cm::CM, res::RES) where {CM <: ComposableModel, RES}
#     _ps = res.p_trained
#     params_traineds = [cm.models[i].st(_ps[cm.param_indices[i]]) for i in 1:length(cm.models)]
#     models = [remake(m, p=params_traineds[i]) for (i,m) in enumerate(models)]
#     return InferenceResult(ComposableModel(models...), res) #/!\ new{Model,RES}( is required! 
# end
