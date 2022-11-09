struct InferenceResult{Model<:AbstractModel,M,P,U0,Pr,R,L}
    model::Model
    minloss::M 
    p_trained::P
    u0s_trained::U0 #  initial condition vector estimated`[u_0_1, ..., u_0_n]`
                 # In the case of independent time series, 
                 # `[[u_0_TS1_1, ..., u_0_TS1_n],...,[u_0_TS1_1,...]]`, matching the format of `res.pred`.
    pred::Pr
    ranges::R
    losses::L
end
import Base.show
Base.show(io::IO, res::InferenceResult) = println(io, "`InferenceResult` with model ", name(res.model))

"""
    $(SIGNATURES)

Container for ouputs of MLE.

# Notes
Uses bijectors to make sure to obtain correct parameter values
"""
function InferenceResult(;model, 
                        minloss = Inf, 
                        p_trained = Float64[], 
                        u0s_trained  = Vector{Float64}[], 
                        pred = Array{Float64,2}[], 
                        ranges = [], 
                        losses = Float64[])
    p_trained = inverse(get_p_bijector(model))(p_trained) # projecting p in true parameter space
    p_tuple = get_re(model)(p_trained) # transforming in tuple
    mp = get_mp(model)
    mp = ParametricModels.remake(mp, p = p_tuple)
    return InferenceResult(SciMLBase.remake(model, mp = mp), 
                        minloss, 
                        p_trained, 
                        u0s_trained,
                        pred,
                        ranges,
                        losses) #/!\ new{Model,RES}( is required! 
end

get_p_trained(res::InferenceResult) = res.model.mp.p

# function construct_result(cm::CM, res::RES) where {CM <: ComposableModel, RES}
#     _ps = res.p_trained
#     params_traineds = [cm.models[i].st(_ps[cm.param_indices[i]]) for i in 1:length(cm.models)]
#     models = [remake(m, p=params_traineds[i]) for (i,m) in enumerate(models)]
#     return InferenceResult(ComposableModel(models...), res) #/!\ new{Model,RES}( is required! 
# end
