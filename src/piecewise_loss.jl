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
function piecewise_loss(infprob::InferenceProblem,
                        θ::AbstractArray{T},
                        ode_data::AbstractArray,
                        tsteps::AbstractArray,
                        ranges::AbstractArray,
                        idx_rngs, 
                        multi_threading=true) where T


    nb_group = length(ranges)
    params = to_param_space(θ, infprob)
    loss_param_prior = get_loss_param_prior(infprob)

    # Calculate multiple shooting loss
    # needs to be of type θ to accept dual numbers
    group_predictions = Vector{Array{T}}(undef, length(ranges))

    # TODO: this could rewritten with inspiration or reuse from "ensemble solve" at:
    # https://github.com/SciML/SciMLBase.jl/blob/master/src/ensemble/basic_ensemble_solve.jl
    if multi_threading
        loss_segments = Vector{T}(undef, length(idx_rngs))
        Threads.@threads for j in 1:length(idx_rngs)
            idx = idx_rngs[j]
            rg = ranges[idx]
            u0_i = _get_u0s(infprob, θ, idx, nb_group)
            l, gp = segment_loss(rg, 
                                u0_i, 
                                infprob,
                                params,
                                ode_data,
                                tsteps)
            if isinf(l)
                return l, group_predictions
            else
                loss_segments[j], group_predictions[j] = l, gp
            end
        end
        loss = sum(loss_segments)
    else
        loss = zero(T)
        for j in 1:length(idx_rngs)
            idx = idx_rngs[j]
            rg = ranges[idx]
            u0_i = _get_u0s(infprob, θ, idx, nb_group)
            l, gp = segment_loss(rg, 
                                u0_i, 
                                infprob,
                                params,
                                ode_data,
                                tsteps)
            if isinf(l)
                return l, group_predictions
            else
                loss += l
                ignore_derivatives() do
                    group_predictions[idx] = gp
                end

            end
        end
    end

    # adding priors
    loss += loss_param_prior(params) # negative log param priors

    return loss, group_predictions
end

function segment_loss(rg, 
                    u0_i,
                    infprob,
                    params,
                    ode_data,
                    tsteps)

    model = get_model(infprob)
    loss_likelihood = get_loss_likelihood(infprob)
    loss_u0_prior = get_loss_u0_prior(infprob)

    data = @view ode_data[:, rg]
    tspan = (tsteps[first(rg)], tsteps[last(rg)])
    tstep_i = @view tsteps[rg]
    sol = simulate(model; u0 = u0_i, tspan = tspan, p = params, saveat = tstep_i)

    # Return infinite loss if one of the integrations failed
    if !(SciMLBase.successful_retcode(sol.retcode))
        ignore_derivatives() do
            @warn "got retcode $(sol.retcode)"
        end
        return Inf, nothing
    else
        pred = sol |> Array
        loss = loss_likelihood(data, pred, tstep_i) # negative loglikelihood
        loss += loss_u0_prior(@view(data[:,1]), u0_i) # negative log u0 priors
        return loss, pred
    end
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