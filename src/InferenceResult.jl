"""
    $(SIGNATURES)

Container for ouputs of MLE.
"""

struct InferenceResult{IP,M,P,U0,Pr,R,L}
    infprob::IP #similar to https://github.com/SciML/SciMLBase.jl/blob/1228b9cf902d005573d20f72372815680d660a67/src/solutions/ode_solutions.jl, we store the initial InferenceProblem
    minloss::M 
    p_trained::P
    u0s_trained::U0 #  initial condition vector estimated`[u_0_1, ..., u_0_n]`
                 # In the case of independent time series, 
                 # `[[u_0_TS1_1, ..., u_0_TS1_n],...,[u_0_TS1_1,...]]`, matching the format of `res.pred`.
    pred::Pr # predicted values, can be stored or not
    ranges::R
    losses::L # vector storing all loss values throughout the iterations, can be stored or not
end
import Base.show
Base.show(io::IO, res::InferenceResult) = println(io, "`InferenceResult` with model ", name(get_model(res.infprob)))

get_p_trained(res::InferenceResult) = res.p_trained