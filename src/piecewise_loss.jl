"""
$(SIGNATURES)

Returns a tuple (`loss`, `pred`) based on the segmentation of `ode_data` 
into segments with time steps given by `tsteps[ranges[i]]`.
The initial conditions are assumed free parameters for each segments.

# Arguments:
  - `infprob`: the inference problem
  - `θ`: [u0,p] where `p` corresponds to the parameters of ode function.
  - `ode_data`: Original Data to be modelled.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `loss_function`: A function to calculate loss, of the form `loss_function(data, params, pred, rg)`
  - `continuity_loss`: Function that takes states ``pred[:,ranges[k][end]]`` and
  ``data[:,ranges[k+1][1]]}`` as input and calculates prediction continuity loss
  between them.
  If no custom `continuity_loss` is specified, `sum(abs, û_end - u_0)` is used.
  - `ranges`: Vector containg range for each segment.
  - `idx_rngs`: Vector containing the indices of the segments to be included in the loss
  - `continuity_term`: Weight term to ensure continuity of predictions throughout
    different groups.

"""
function piecewise_loss(
    infprob::InferenceProblem,
    θ::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    loss_function,
    continuity_loss,
    ranges::AbstractArray,
    idx_rngs;
    continuity_term::Real=0
)
    model = get_model(infprob)
    dim_prob = get_dims(model)
    nb_group = length(ranges)
    @assert length(θ) > nb_group * dim_prob "`params` should contain [u0;p]"

    params = _get_param(infprob, θ, nb_group) # params of the problem

    # Calculate multiple shooting loss
    loss = zero(eltype(θ))
    group_predictions = Vector{Array{eltype(θ)}}(undef, length(ranges))
    for i in idx_rngs
        rg = ranges[i]
        u0_i = _get_u0s(infprob, θ, i, nb_group) # taking absolute value, assuming populations cannot be negative
        data = ode_data[:, rg]
        tspan = (tsteps[first(rg)], tsteps[last(rg)])
        sol = simulate(model; u0 = u0_i, tspan = tspan, p = params, saveat = tsteps[rg])
        # Abort and return infinite loss if one of the integrations failed
        sol.retcode == :Success && sol.retcode !== :Terminated ? nothing : return Inf, group_predictions

        pred = sol |> Array
        loss += loss_function(data, params, pred, rg)
        group_predictions[i] = pred

        if i < nb_group && continuity_term > 0.0
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss +=
                continuity_term * continuity_loss(pred[:, end], ode_data[:, first(ranges[i+1])])
        end
    end

    return loss, group_predictions
end

function piecewise_loss(
    infprob::InferenceProblem,
    θ::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    loss_function::Function,
    ranges::AbstractArray,
    idx_rngs;
    kwargs...
)

    return piecewise_loss(
        infprob,
        θ,
        ode_data,
        tsteps,
        loss_function,
        _default_continuity_loss,
        ranges,
        idx_rngs;
        kwargs...
    )
end

# Default ontinuity loss between last state in previous prediction
# and current initial condition in ode_data
function _default_continuity_loss(û_end::AbstractArray,
    u_0::AbstractArray)
    return mean((û_end - u_0) .^ 2)
end

# projecting θ in optimization space to param in Tuple form in true parameter space
function _get_param(infprob::InferenceProblem, θ, nb_group)
    dim_prob = get_dims(infprob)
    p̃ = @view θ[nb_group*dim_prob+1:end]
    # projecting p in true parameter space
    p = inverse(get_p_bijector(infprob))(p̃) 
    # converting to named tuple, for easy handling
    p_tuple = get_re(infprob)(p)
    return p_tuple
end

# projecting θ in optimization space to u0 for segment i in true parameter space
function _get_u0s(infprob::InferenceProblem, θ, i, nb_group)
    dim_prob = get_dims(infprob)
    @assert 0 < i <= nb_group "trying to access undefined segment"
    ũ0 = @view θ[dim_prob*(i-1)+1:dim_prob*i]
    # projecting in true parameter space
    u0 = inverse(get_u0_bijector(infprob))(ũ0)
    return u0
end