function loglikelihood(data::Matrix, tsteps, ranges, infprob::InferenceProblem, pflat::Vector, u0s::Vector)

    # projecting p and u0s in parameter space, 
    # to further use `piecewise_loss`
    p_bij = get_p_bijector(infprob)
    θ_p = p_bij(pflat)
    θ_u0s =  [get_u0_bijector(infprob)(u0) for u0 in u0s]
    θ_u0s = vcat(θ_u0s...)
    θ = [θ_u0s; θ_p]

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

function loglikelihood(data::Matrix, tsteps, ranges, infprob::InferenceProblem, p::NamedTuple, u0s::Vector)
    p_flat, _ = destructure(p)
    loglikelihood(data, tsteps, ranges, infprob, p_flat, u0s)
end

"""
    Provides evidence `P(M|data)` for model `M` stored in `infprob` given the `data`, 
    using MAP estimate `p` and `u0s`.

    Relies on [Laplace's method](https://en.wikipedia.org/wiki/Laplace's_method)
"""
function get_evidence(data::Matrix, tsteps, ranges, infprob::InferenceProblem, p::NamedTuple, u0s::Vector)
    # To be completed
    p_flat, _ = destructure(p)
    ll(p) = loglikelihood(data, tsteps, ranges, infprob, p, u0s)
    A = - ForwardDiff.hessian(ll, p_flat)
    ll_map = loglikelihood(data, tsteps, ranges, infprob, p_flat, u0s)
    return ll_map - log(det(A / (2π)))
end