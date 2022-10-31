"""
$(SIGNATURES)

Returns a tuple (`loss`, `pred`) obtained from piecewiseing of the 
time series `ode_data` into segments with time steps given by `tsteps[ranges[i]]`.
The initial conditions are assumed free parameters for each segments.
! the dynamics is assumed to lie in R⁺ !

# Arguments:
  - `θ`: [u0,p] where `p` corresponds to the parameters of ode function.
  - `ode_data`: Original Data to be modelled.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `model`: ODE model
  - `loss_function`: A function to calculate loss, of the form `loss_function(data, params, pred, rg)`
  - `continuity_loss`: Function that takes states ``pred[:,ranges[k][end]]`` and
  ``data[:,ranges[k+1][1]]}`` as input and calculates prediction continuity loss
  between them.
  If no custom `continuity_loss` is specified, `sum(abs, û_end - u_0)` is used.
  - `ranges`: Vector containg range for each segment.
  - `continuity_term`: Weight term to ensure continuity of predictions throughout
    different groups.
# Note:
The parameter 'continuity_term' should be a relatively big number to enforce a large penalty
whenever the last point of any group doesn't coincide with the first point of next group.
"""
function piecewise_loss(
    θ::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    model::AbstractModel,
    loss_function,
    continuity_loss,
    ranges::AbstractArray;
    continuity_term::Real=0,
    kwargs...
)
    dim_prob = get_dims(model)
    nb_group = length(ranges)
    @assert length(θ) > nb_group * dim_prob "`params` should contain [u0;p]"

    params = _get_param(θ, nb_group, dim_prob) # params of the problem
    u0s = _get_u0s(θ, nb_group, dim_prob)

    # Calculate multiple shooting loss
    loss = zero(eltype(θ))
    group_predictions = Vector{Array{eltype(θ)}}(undef, length(ranges))
    for (i, rg) in enumerate(ranges)
        u0_i = u0s[i] # taking absolute value, assuming populations cannot be negative
        data = ode_data[:, rg]
        sol = simulate(model;
                        u0 = u0_i,
                        tspan = (tsteps[first(rg)], tsteps[last(rg)]), 
                        p = params, 
                        saveat = tsteps[rg], 
                        kwargshandle = KeywordArgError)
        # Abort and return infinite loss if one of the integrations failed
        sol.retcode == :Success && sol.retcode !== :Terminated ? nothing : return Inf, group_predictions

        pred = sol |> Array
        loss += loss_function(data, params, pred, rg)
        group_predictions[i] = pred

        if i < nb_group && continuity_term > 0.
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss +=
                continuity_term * continuity_loss(pred[:, end], ode_data[:, first(ranges[i+1])])
        end
    end

    return loss, group_predictions
end

function piecewise_loss(
    θ::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    model::AbstractModel,
    loss_function::Function,
    ranges::AbstractArray;
    kwargs...,
)

    return piecewise_loss(
            θ,
            ode_data,
            tsteps,
            model,
            loss_function,
            _default_continuity_loss,
            ranges::AbstractArray;
            kwargs...,
        )
end

# Default ontinuity loss between last state in previous prediction
# and current initial condition in ode_data
function _default_continuity_loss(û_end::AbstractArray,
    u_0::AbstractArray)
    return mean((û_end - u_0).^2)
end

function _get_param(θ, nb_group, dim_prob)
    return @view θ[nb_group * dim_prob + 1: end]
end

function _get_u0s(θ, nb_group, dim_prob)
    # @show nb_group, dim_prob
    return [abs.(θ[dim_prob*(i-1)+1:dim_prob*i]) for i in 1:nb_group]
end