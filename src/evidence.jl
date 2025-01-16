"""
$SIGNATURES

Get loglikelihood of `infprob` evaluated at parameters `p` and ICs `u0s`.
`tsteps, ranges` are required to recover the segments used.
"""

function loglikelihood(data::Matrix, tsteps, infprob::InferenceProblem, p::AbstractVector, u0s::AbstractVector, ranges::AbstractVector)
    θ = _build_θ(p, u0s, infprob)

    idx_rngs = 1:length(ranges)

    # using `piecewise_loss`
    ll, _ = piecewise_loss(infprob,
                        θ, 
                        data, 
                        tsteps, 
                        ranges,
                        idx_rngs)
    # `piecewise_loss` is the negative of the loglikelihood
    return - ll
end

function loglikelihood(data::Matrix, tsteps, infres::InferenceResult, p::AbstractVector)
    loglikelihood(data, tsteps, infres.infprob, p, infres.u0s_trained, infres.ranges)
end

# """
# $SIGNATURES

# Provides evidence `p(M|data) = ∫p(data|M, θ) p(θ) p(M) dθ` for model `M` stored in `infprob` given the `data`, 
# using MAP estimate `p` and `u0s`. Here it is assumed that `p(M) = 1`.

# Relies on [Laplace's method](https://en.wikipedia.org/wiki/Laplace's_method)

# # Note
# For now, we do not integrate over initial conditions `u0s`, but this may be considered.
# """
# function get_evidence(data::Matrix, tsteps, infprob::InferenceProblem, p::ComponentArray, u0s::AbstractVector, ranges::AbstractVector)
#     ll(p) = loglikelihood(data, tsteps, infprob, p, u0s, ranges)
#     A = - ForwardDiff.hessian(ll, p)
#     ll_map = loglikelihood(data, tsteps, infprob, p, u0s, ranges)
#     return ll_map - log(det(A / (2π)))
# end

# function get_evidence(data::Matrix, tsteps, infres::InferenceResult)
#     get_evidence(data, tsteps, infres.infprob, infres.p_trained, infres.u0s_trained, infres.ranges)
# end