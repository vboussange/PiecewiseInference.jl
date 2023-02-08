"""
$(SIGNATURES)

Returns a tuple (`loss`, `pred`) based on the segmentation of `ode_data` 
into segments with time steps given by `tsteps[ranges[i]]`.
The initial conditions are assumed free parameters for each segments.

# Arguments:
  - `infprob`: the inference problem
  - `θ`: [u0,p] where `p` corresponds to the parameters of ode function in the optimization space.
  - `ode_data`: Original Data to be modeloss_likelihooded.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `ranges`: Vector containg range for each segment.
  - `idx_rngs`: Vector containing the indices of the segments to be included in the loss
"""
function piecewise_loss(
                        infprob::InferenceProblem,
                        θ::AbstractArray,
                        ode_data::AbstractArray,
                        tsteps::AbstractArray,
                        ranges::AbstractArray,
                        idx_rngs)

    model = get_model(infprob)
    dim_prob = get_dims(model)
    nb_group = length(ranges)
    loss_likelihood = get_loss_likelihood(infprob)
    loss_u0_prior = get_loss_u0_prior(infprob)
    loss_param_prior = get_loss_param_prior(infprob)

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
        loss += loss_likelihood(data, pred, rg) # negative loglikelihood
        loss += loss_u0_prior(data[:,1], u0_i) # negative log u0 priors
        group_predictions[i] = pred

    end
    # adding priors
    loss += loss_param_prior(params) # negative log param priors

    return loss, group_predictions
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