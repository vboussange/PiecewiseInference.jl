"""
Returns a tuple (`loss`, `pred`) obtained from minibatching of the 
time series `ode_data` into segments with time steps given by `tsteps[ranges[i]]`.
The initial conditions are assumed free parameters for each segments.
Arguments:
  - `θ`: [u0,p] where `p` corresponds to the parameters of ode function.
  - `ode_data`: Original Data to be modelled.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `prob`: ODE problem that the Neural Network attempts to solve.
  - `loss_function`: Any arbitrary function to calculate loss.
  - `continuity_loss`: Function that takes states ``\\hat{u}_{end}`` of group ``k`` and
  ``u_{0}`` of group ``k+1`` as input and calculates prediction continuity loss
  between them.
  If no custom `continuity_loss` is specified, `sum(abs, û_end - u_0)` is used.
  - `solver`: ODE Solver algorithm.
  - `ranges`: Vector containg range for each segment.
  - `continuity_term`: Weight term to ensure continuity of predictions throughout
    different groups.
  - `kwargs`: Additional arguments splatted to the ODE solver. Refer to the
  [Local Sensitivity Analysis](https://diffeq.sciml.ai/dev/analysis/sensitivity/) and
  [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
  documentation for more details.
Note:
The parameter 'continuity_term' should be a relatively big number to enforce a large penalty
whenever the last point of any group doesn't coincide with the first point of next group.
"""
function minibatch_loss(
    θ::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    prob::ODEProblem,
    loss_function,
    continuity_loss,
    solver::DiffEqBase.AbstractODEAlgorithm,
    ranges::AbstractArray;
    continuity_term::Real=100,
    kwargs...
)
    dim_prob = length(prob.u0)
    nb_group = length(ranges)
    @assert length(θ) > nb_group * dim_prob "`params` should contain [u0;p]"

    params = @view θ[nb_group * dim_prob + 1: end] # params of the problem

    # Calculate multiple shooting loss
    loss = zero(eltype(θ))
    group_predictions = Vector{Array{eltype(θ)}}(undef, length(ranges))
    for (i, rg) in enumerate(ranges)
        u0_i = @view θ[dim_prob*(i-1)+1:dim_prob*i]
        prob_i = remake(prob; p=params, tspan=(tsteps[first(rg)], tsteps[last(rg)]), u0=u0_i,)
        u = ode_data[:, rg]
        sol = solve(prob_i, solver; saveat=tsteps[rg], kwargshandle=KeywordArgError, kwargs...)

        # Abort and return infinite loss if one of the integrations failed
        sol.retcode == :Success ? nothing : return Inf, group_predictions

        û = sol |> Array
        loss += loss_function(u, û)
        Zygote.ignore() do # this relates to issue https://github.com/FluxML/Zygote.jl/issues/571
            group_predictions[i] = û
        end
        if i < nb_group && continuity_term > 0.
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss +=
                continuity_term * continuity_loss(û[:, end], ode_data[:, first(ranges[i+1])])
        end
    end

    return loss, group_predictions
end

function minibatch_loss(
    θ::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    prob::ODEProblem,
    loss_function::Function,
    solver::DiffEqBase.AbstractODEAlgorithm,
    ranges::AbstractArray;
    kwargs...,
)

    return minibatch_loss(
            θ,
            ode_data,
            tsteps,
            prob,
            loss_function,
            _default_continuity_loss,
            solver,
            ranges::AbstractArray;
            kwargs...,
        )
end

# Default ontinuity loss between last state in previous prediction
# and current initial condition in ode_data
function _default_continuity_loss(û_end::AbstractArray,
    u_0::AbstractArray)
    return sum(abs, û_end - u_0)
end