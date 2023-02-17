"""
$SIGNATURES

Get loglikelihood of `infprob` evaluated at parameters `p` and ICs `u0s`.
`tsteps, ranges` are required to recover the segments used.
"""
function loglikelihood(data::Matrix, tsteps, infprob::InferenceProblem, p::ComponentArray, u0s::Vector; kwargs...)
    # projecting p and u0s in parameter space, 
    # to further use `piecewise_loss`
    datasize = size(data,2)
    ranges = get_ranges(; datasize, kwargs...)
    θ = _build_θ(p, get_p_bijector(infprob), u0s, get_u0_bijector(infprob))

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

"""
$SIGNATURES

Provides evidence `p(M|data) = ∫p(data|M, θ) p(θ) p(M) dθ` for model `M` stored in `infprob` given the `data`, 
using MAP estimate `p` and `u0s`. Here it is assumed that `p(M) = 1`.

Relies on [Laplace's method](https://en.wikipedia.org/wiki/Laplace's_method)

# Note
For now, we do not integrate over initial conditions `u0s`, but this may be considered.
"""
function get_evidence(data::Matrix, tsteps, infprob::InferenceProblem, p::ComponentArray, u0s::Vector; kwargs...)
    ll(p) = loglikelihood(data, tsteps, infprob, p, u0s; kwargs...)
    A = - ForwardDiff.hessian(ll, p)
    ll_map = loglikelihood(data, tsteps, infprob, p, u0s; kwargs...)
    return ll_map - log(det(A / (2π)))
end