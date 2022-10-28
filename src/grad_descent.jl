#=
utils for gradient descent alg.
=#
"""
$(SIGNATURES)

## Arguments
- `minlog`: box constraint min, logscale
- `maxlog`: box constraint max, log scale.
"""
function loss_log(data, params, pred, rg, ic_term, padding_data, prior_scaling; minlog = -Inf, maxlog=Inf)
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    l = mean((log.(data[padding_data[:,rg]]) - log.(pred[padding_data[:,rg]])).^2)
    l += mean((log.(data[:,1][padding_data[:,rg[1]]]) - log.(pred[:,1][padding_data[:,rg[1]]])).^2) * ic_term # putting more weights on initial conditions
    l += mean((abs.(params[1:size(data,1)]) .- 0.1).^2) * prior_scaling # prior on r, to force it to be > 0.
    ## adding an extra term to constrain the unrealistic prediction for unconstrained data
    # idx_range = minlog .< log.(pred[:,1]) .< maxlog
    # if any(.! idx_range)
    #     # @show log.(pred[:,1])[1]
    #     # @show rg
    #     # @show count(.! idx_range)
    #     meanlog = (minlog .+ maxlog) / 2
    #     extraterm = (log.(pred[.! idx_range, 1]) - meanlog[.! idx_range]).^2
    #     l += mean(extraterm)
    # end
    if l isa Number # preventing any other reason for Zygote to crash
        return l
    else 
        return Inf
    end
end