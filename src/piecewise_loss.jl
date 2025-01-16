"""
$(SIGNATURES)

Returns a tuple (`loss`, `pred`) based on the segmentation of `ode_data` into
segments with time steps given by `tsteps[ranges[i]]`. The initial conditions
are assumed free parameters for each segments.

# Arguments:
  - `infprob`: the inference problem
  - `θ`: [u0,p] where `p` corresponds to the parameters of ode function in the
    optimization space.
  - `ode_data`: Original Data to be modeloss_likelihooded.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `ranges`: Vector containg range for each segment.
  - `idx_rngs`: Vector containing the indices of the segments to be included in
    the loss. Useful for minibatching
"""
function piecewise_loss(infprob::InferenceProblem,
                        θ::AbstractArray{T},
                        ode_data::AbstractArray,
                        tsteps::AbstractArray,
                        ranges::AbstractArray,
                        idx_rngs, 
                        multi_threading=true) where T

    model = get_model(infprob)
    nb_group = length(ranges)
    loss_likelihood = get_loss_likelihood(infprob)
    loss_u0_prior = get_loss_u0_prior(infprob)
    loss_param_prior = get_loss_param_prior(infprob)

    params = to_param_space(θ, infprob)

    # Calculate multiple shooting loss
    # needs to be of type θ to accept dual numbers
    group_predictions = Vector{Array{T}}(undef, length(ranges))

    if multi_threading
        loss_vec = Vector{T}(undef, length(idx_rngs))
        Threads.@threads for i in 1:length(idx_rngs)
            idx = idx_rngs[i]
            rg = ranges[idx]
            u0_i = _get_u0s(infprob, θ, idx, nb_group)
            tstep_i = @view tsteps[rg]
            data = @view ode_data[:, rg]
            tspan = (tsteps[first(rg)], tsteps[last(rg)])
            sol = simulate(model; u0 = u0_i, tspan = tspan, p = params, saveat = tsteps[rg])

            # Abort and return infinite loss if one of the integrations failed
            if !(SciMLBase.successful_retcode(sol.retcode))
                ignore_derivatives() do
                    @warn "got retcode $(sol.retcode)"
                end
                return Inf, group_predictions
            end

            pred = sol |> Array
            loss_vec[i] = loss_likelihood(data, pred, tstep_i) # negative loglikelihood
            loss_vec[i] += loss_u0_prior(@view(data[:,1]), u0_i) # negative log u0 priors

            # used for plotting, no need to differentiate
            ignore_derivatives() do
                group_predictions[i] = pred
            end
        end
        loss = sum(loss_vec)
    else
        loss = zero(T)
        for i in idx_rngs
            rg = ranges[i]
            u0_i = _get_u0s(infprob, θ, i, nb_group)
            tstep_i = @view tsteps[rg]
            data = @view ode_data[:, rg]
            tspan = (tsteps[first(rg)], tsteps[last(rg)])
            sol = simulate(model; u0 = u0_i, tspan = tspan, p = params, saveat = tsteps[rg])

            # Abort and return infinite loss if one of the integrations failed
            if :retcode in fieldnames(typeof(sol))
                if !(SciMLBase.successful_retcode(sol.retcode))
                    ignore_derivatives() do
                        @warn "got retcode $(sol.retcode)"
                    end
                    return Inf, group_predictions
                end
            end

            pred = sol |> Array
            loss += loss_likelihood(data, pred, tstep_i) # negative loglikelihood
            loss += loss_u0_prior(data[:,1], u0_i) # negative log u0 priors

        end
    end

    loss += loss_param_prior(params) # negative log param priors

    return loss
end

# projecting θ in optimization space to u0 for segment i in true parameter space
function _get_u0s(infprob::InferenceProblem, θ, i, nb_group)
    dim_prob = get_dims(infprob)
    @assert 0 < i <= nb_group "trying to access undefined segment"
    ũ0 = @view θ.u0s[dim_prob*(i-1)+1:dim_prob*i]
    # projecting in true parameter space
    u0 = inverse(get_u0_bijector(infprob))(ũ0)
    return u0
end