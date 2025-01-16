"""
$(SIGNATURES)
Get ranges that partition data of length `datasize` in groups of `groupsize` observations.
If the data isn't perfectly dividable by `groupsize`, the last group contains
the reminding observations.
Taken from https://github.com/SciML/DiffEqFlux.jl/blob/80c4247c19860d0422211d6a65283d896eeaa831/src/multiple_shooting.jl#L273-L303
```julia
group_ranges(datasize, groupsize)
```
Arguments:
- `datasize`: amount of data points to be partitioned
- `groupsize`: maximum amount of observations in each group
Example:
```julia-repl
julia> group_ranges(10, 5)
3-element Vector{UnitRange{Int64}}:
 1:5
 5:9
 9:10
```
"""
function group_ranges_gs(datasize::Integer, group_size::Integer)
    2 <= group_size <= datasize || throw(
        DomainError(
            group_size,
            "datasize must be positive and group_size must to be within [2, datasize]",
        ),
    )
    return [i:min(datasize, i + group_size - 1) for i in 1:group_size-1:datasize-1]
end

"""
$(SIGNATURES)

    Similar to `group_ranges`, except that it takes as arguments the nb of groups wanted, `group_nb`
"""
function group_ranges_gn(datasize::Integer, group_nb::Integer)
    group_size = ceil(Integer, datasize/group_nb) + 1
    group_ranges_gs(datasize, group_size)
end

"""
$SIGNATURES

Returns the loglikelihood of `params` given the prior distribution of the parameters `param_distrib`
- `params`: params, in the form of NamedTuple
- `param_distrib`: in the form of a `Dictionary` or a `NamedTuple`, with entries `p::String` => "d::Distribution"
"""
function loss_param_prior_from_dict(params, param_distrib)
    l = 0.
    # parameter prior
    for k in keys(param_distrib)
        l += logpdf(param_distrib[k], reshape(params[k],:))
    end
    return - l
end

function to_optim_space(p::ComponentArray, infprob::InferenceProblem)
    @unpack p0, p_bij = infprob
    pairs = [reshape(p_bij[k](getproperty(p,k)),:) for k in keys(p0)]
    ax = getaxes(p0)
    return ComponentArray(vcat(pairs...), ax)
end

# TODO /!\ order is not guaranteed!
function to_param_space(θ, infprob::InferenceProblem)
    @unpack p0, p_bij = infprob
    pairs = [reshape(inverse(p_bij[k])(getproperty(θ,k)),:) for k in keys(p0)]
    ax = getaxes(p0)
    return ComponentArray(vcat(pairs...), ax)
end