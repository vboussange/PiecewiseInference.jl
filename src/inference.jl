"""
$(SIGNATURES)

Similar to `inference` but for independent time series, where `data`
is a vector containing the independent arrays corresponding to the time series,
and `tsteps` is a vector where each entry contains the time steps
of the corresponding time series.
"""
function piecewise_ML_indep_TS(infprob;
                                data,
                                group_size = nothing, 
                                group_nb = nothing,
                                tsteps::Vector, #corresponding time steps
                                save_losses = true, # saving prediction
                                kwargs...)
    @assert length(tsteps) == length(data) "Independent time series must be gathered as a Vector"
    @assert all(size(data[1],1) .== size.(data, 1)) "Independent time series must have same state variable dimension"

    datasize_arr = size.(data,2)
    ranges_arr = [get_ranges(;group_size, group_nb, datasize = datasize_arr[i]) for i in 1:length(data)]
    # updating to take into account the shift provoked by concatenating independent TS
    ranges_shift = cumsum(datasize_arr) # shift
    for i in 2:length(ranges_arr)
        for j in 1:length(ranges_arr[i]) # looping through rng in each independent TS
            ranges_arr[i][j] = ranges_shift[i-1] .+ ranges_arr[i][j] #adding shift to the start of the range
        end
    end
    data_cat = cat(data...,dims=2)
    ranges_cat = vcat(ranges_arr...)
    tsteps_cat = vcat(tsteps...)

    res = inference(infprob;
                        ranges=ranges_cat,
                        data=data_cat, 
                        tsteps=tsteps_cat, 
                        kwargs...) 
                        # this overrides kwargs, essential as it does not 
                        # make sense to have continuity across indepdenent TS
                        # NOTE: we could have continuity within a time series, 
                        # this must be carefully thought out.
        
    # reconstructing the problem with original format
    ranges_arr = [get_ranges(;group_size, group_nb, datasize = datasize_arr[i]) for i in 1:length(data)]
    idx_res = [0;cumsum(length.(ranges_arr))]

    # group u0s in vector of u0s, 
    # [[u_0_TS1_1, ..., u_0_TS1_n],...,[u_0_TS1_1,...]]
    u0s_trained_arr = [res.u0s_trained[idx_res[i]+1:idx_res[i+1]] for i in 1:length(data)]
    
    save_losses ? losses = res.losses : losses = nothing

    res_arr = InferenceResult(infprob,
                            res.minloss,
                            res.p_trained,
                            u0s_trained_arr, 
                            ranges_arr, 
                            losses,)
    return res_arr
end

"""
$(SIGNATURES) performs piecewise inference for a given `InferenceProblem` and
`data`. Loops through the optimizers `optimizers`. Returns a `InferenceResult`.
# Arguments
- `infprob`:  An instance of `InferenceProblem` that defines the model, the
  parameter constraints and its likelihood function.
- `opt` : An array of optimizers that will be used to maximize the likelihood
  (minimize the loss).
- `group_size` : The size of the segments. It is an alternative to `group_nb`,
  and specifies the number of data point in each segment.
- `group_nb`: Alternatively to `group_size`, one can ask for a certain number of
  segments.
- `ranges`: Alternatively to `group_size` and `group_nb`, one can directly
  provide a vector of indices, where each entry corresponds to the indices of a
  segment. Possibly provided by `get_ranges`. 
- `data` : The data to fit.
- `tsteps` : The time steps for which the data was recorded.
# Optional
- `u0_init` : A vector of initial guesses for the initial conditions for each
  segment. If not provided, initial guesses are initialized from the `data`.
- `optimizers` : array of optimizers, e.g. `[Adam(0.01)]`
- `epochs` : A vector with number of epochs for each optimizer in `optimizers`.
- `batchsizes`: A vector of batch sizes, which should match the length of
  `optimizers`. If nothing is provided, all segments are used at once (full
  batch).
- `verbose_loss` : Whether to display loss during training.
- `info_per_its = 50`: The frequency at which to display the training
  information.
- `cb` :  A call back function. Must be of the form `cb(p, u0s, l)`.
- `save_losses = true` : Whether to save the losses.
- `adtype = Optimization.AutoForwardDiff()` : The automatic differentiation (AD)
  type to be used. Can be `Optimization.AutoForwardDiff()` for forward AD or
 `Optimization.Autozygote()` for backward AD.
- `u0s_init = nothing`: if provided, should be a vector of the form `[u0_1, ...,
  u0_n]` where `n` is the number of segments
- `multi_threading = true`: if `true`, segments in the piecewise loss are
computed in parallel. Currently not supported with `adtype =
Optimization.Autozygote()` 
# Examples
```julia
using SciMLSensitivity # provides diffential equation sensitivity methods
using UnPack # provides the utility macro @unpack 
using OptimizationOptimisers # provide the optimizers
using LinearAlgebra
using PiecewiseInference
using OrdinaryDiffEq
using Distributions, Bijectors # used to constrain parameters and initial conditions
@ODEModel MyModel
function (m::MyModel)(du, u, p, t)
    @unpack b = p
    du .=  0.1 .* u .* ( 1. .- b .* u) 
end
tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])
p_true = (b = [0.23, 0.5],)
p_init= (b = [1., 2.],)
# Defining the model
# Pay attention to the semi column before the parameters for `ModelParams`
u0 = ones(2)
mp = ModelParams(;p = p_true, 
                tspan,
                u0, 
                alg = BS3(),
                sensealg = ForwardDiffSensitivity(),
                saveat = tsteps, 
                )
model = MyModel(mp)
sol_data = simulate(model)
ode_data = Array(sol_data)
# adding some normally distributed noise
σ_noise = 0.1
ode_data_wnoise = ode_data .+ randn(size(ode_data)) .* σ_noise
# Define the `InferenceProblem`
# First specifiy which values can the parameter take with bijectors
# here, `b` is constrained to be ∈ [1e-3, 5e0] and `u0` ∈ [1e-3, 5.]
p_bij = (bijector(Uniform(1e-3, 5e0)),)
u0_bij = bijector(Uniform(1e-3,5.))
distrib_noise = MvNormal(ones(2) * σ_noise^2)
# defining `loss_likelihood`
loss_likelihood(data, pred, tsteps) = sum(logpdf(distrib_noise, data .- pred))
infprob = InferenceProblem(model, p_init; p_bij, u0_bij)
optimizers = [OptimizationOptimisers.Adam(0.001)]
epochs = [5000]
group_nb = 2
batchsizes = [1] # batch size used for each optimizer in optimizers (here only one)
# you could also have `batchsizes = [group_nb]`
res = inference(infprob,
                    group_nb = group_nb, 
                    data = ode_data_wnoise, 
                    tsteps = tsteps, 
                    epochs = epochs, 
                    optimizers = optimizers,
                    batchsizes = batchsizes,
                    )
p_trained = get_p_trained(res)
pred = res.pred
"""
function inference(infprob;
                    data,
                    tsteps,
                    group_size = nothing, 
                    group_nb = nothing,
                    ranges = nothing, # provided by `inference`
                    optimizers = [OptimizationOptimisers.Adam(0.01), OptimizationOptimJL.BFGS(initial_stepnorm=0.01)],
                    epochs = [1000, 200],
                    batchsizes = nothing,
                    verbose_loss = true,
                    info_per_its=50,
                    cb = nothing,
                    save_losses = true,
                    u0s_init = nothing,
                    adtype = Optimization.AutoForwardDiff(),
                    multi_threading=false)
    model = get_model(infprob)
    dim_prob = get_dims(model) #used by loss_nm

    # generating segment time indices (`ranges`) from kwargs
    if (isnothing(group_size) + isnothing(group_nb) + isnothing(ranges)) == 0
        throw(ArgumentError("Need to provide `group_size`, `group_nb` or `ranges`"))
    elseif (!isnothing(group_size) + !isnothing(group_nb) + !isnothing(ranges)) > 1
        throw(ArgumentError("Cannot handle combinations of `group_size`, `group_nb` or `ranges`. Choose only one keyword"))
    elseif isnothing(ranges)
        datasize = size(data,2)
        ranges = get_ranges(;group_size, group_nb, datasize)
    end

    idx_ranges = 1:length(ranges) # idx of batches

    isnothing(batchsizes) && (batchsizes = fill(length(ranges),length(epochs)))

    @assert (length(optimizers) == length(epochs) == length(batchsizes)) "`optimizers`, `epochs`, `batchsizes` must be of same length"
    @assert ((size(data,1) == dim_prob) || !isnothing(u0s_init)) "The dimension of the training data does not correspond to the dimension of the state variables. This probably means that the training data corresponds to observables different from the state variables. In this case, you need to provide manually `u0s_init`." 
    for (i,opt) in enumerate(optimizers)
        OPT = typeof(opt)
        if OPT <: Optim.AbstractOptimizer
            @assert batchsizes[i] == length(idx_ranges) "$OPT is not compatible with mini-batches - use `batchsizes = group_nb`"
        end
    end
    p0 = get_p(infprob)
    @assert all([p0[k] isa AbstractArray for k in keys(p0)]) "Each values of `p0` must be `Array`s"

    # initialising with data if not provided
    if isnothing(u0s_init) 
        u0s_init = _init_u0s(data, ranges)
    end
    # build θ, which is the parameter vector containing u0s, in the parameter space

    θ_0 = _build_θ(p0, u0s_init, infprob)

    # piecewise loss
    function _loss(θ, batch_idx_ranges)
        return piecewise_loss(infprob,
                            θ, 
                            data, 
                            tsteps, 
                            ranges,
                            batch_idx_ranges,
                            multi_threading)
    end

    nb_group = length(ranges)
    println("inference with $(length(tsteps)) points and $nb_group groups.")

    # Here we need a default behavior for Optimization.jl (see https://github.com/SciML/Optimization.jl/blob/c0a51120c7c54a89d091b599df30eb40c4c0952b/lib/OptimizationFlux/src/OptimizationFlux.jl#L53)
    callback(state, l) = begin
        push!(losses, l)
        # TODO: this is called at every parameter update, it would be nice to have it at every epoch
        if length(losses)%info_per_its==0
            verbose_loss && (println("Loss after $(length(losses)) iterations: $(losses[end])"))
        end
        if !isnothing(cb)
            p = to_param_space(state.u, infprob)
            u0s = [_get_u0s(infprob, state.u, i, nb_group) for i in 1:nb_group]
            cb(p, u0s, l)
        end
        return false
    end

    ################
    ### TRAINING ###
    ################
    # Container to track the losses
    losses = eltype(data)[]

    @info "Training started"
    objectivefun = OptimizationFunction(_loss, adtype) # similar to https://sensitivity.sciml.ai/stable/ode_fitting/stiff_ode_fit/
    opt = first(optimizers)
    θ = __solve(opt, objectivefun, θ_0, idx_ranges, first(batchsizes), first(epochs), callback)
    for (i, opt) in enumerate(optimizers[2:end])
        θ = __solve(opt, objectivefun, θ, idx_ranges, batchsizes[i+1], epochs[i+1], callback)
    end
    
    
    minloss = _loss(θ, idx_ranges)
    p_trained = to_param_space(θ, infprob)

    u0s_trained = [_get_u0s(infprob, θ, i, nb_group) for i in 1:nb_group]

    @info "Minimum loss for all batches: $minloss"
    
    save_losses ? nothing : losses = nothing
    res = InferenceResult(infprob,
                            minloss, 
                            p_trained,
                            u0s_trained,
                            ranges, 
                            losses,)
    return res
end

"""
$(SIGNATURES)
"""
function get_ranges(;group_size = nothing, group_nb = nothing, datasize)

    if !isnothing(group_size)
        if group_size-1 < datasize
            ranges = group_ranges_gs(datasize, group_size)
            # piecewiseing
        else
            ranges = [1:datasize]
            # normal MLE with initial estimation
        end
        return ranges
    elseif !isnothing(group_nb)
        if group_nb > 1
            ranges = group_ranges_gn(datasize, group_nb)
            # piecewiseing
        else
            ranges = [1:datasize]
            # normal MLE with initial estimation
        end
        return ranges
    else 
        ArgumentError("Need to provide whether `group_nb` or `group_size` as keyword argument")
    end
    
end

# This supports batching
function __solve(opt::OPT, objectivefun, θ_0, idx_ranges, batchsize, epochs, callback) where OPT <: Optimisers.AbstractRule
    train_loader = MLUtils.DataLoader(idx_ranges; batchsize, shuffle = true, partial=true)
    optprob = Optimization.OptimizationProblem(objectivefun, θ_0, train_loader)
    @info "Running optimizer $(OPT)"
    res = Optimization.solve(optprob,
                            opt;
                            epochs=epochs,
                            callback=callback)
    return res.u
end

# this does not support batching, batchsizes
function __solve(opt::OPT, objectivefun, θ_0, idx_ranges, batchsize, epochs, callback) where OPT <: Optim.AbstractOptimizer
    @assert batchsize == length(idx_ranges)
    objectivefun_nobatch = OptimizationFunction((u, p) -> objectivefun.f(u, idx_ranges), objectivefun.adtype)
    optprob = Optimization.OptimizationProblem(objectivefun_nobatch, θ_0, nothing)
    
    @info "Running optimizer $(OPT)"
    res = Optimization.solve(optprob,
                            opt;
                            maxiters=epochs,
                            callback=callback)
    return res.u
end

# DEPRECATED
# """
# $(SIGNATURES)

# Performs a iterative piecewise MLE, iterating over `group_sizes`. 
# Stops the iteration when loss function increases between two iterations.

# Returns an array with all `InferenceResult` obtained during the iteration.
# For kwargs, see `inference`.

# # Note 
# - for now, does not support independent time series (`piecewise_ML_indep_TS`).
# - at every iteration, initial conditions are initialised given the predition of previous iterations

# # Specific arguments
# - `group_sizes` : array of group sizes to test
# - `optimizers_array`: optimizers_array[i] is an array of optimizers for the trainging process of `group_sizes[i]`
# """
# function iterative_inference(infprob;group_sizes = nothing,
#                                 group_nbs = nothing,
#                                 optimizers_array,
#                                 threshold = 1e-16,
#                                 kwargs...)

#     @assert length(group_sizes) == length(optimizers_array)

#     # initialising results
#     data = kwargs[:data]
#     datasize = size(data,2)
#     p_trained = get_p(infprob)
#     res = InferenceResult(infprob,
#                         Inf,
#                         p_trained,
#                         Vector{Float64}[],
#                         [data], 
#                         [1:datasize],
#                         Float64[])
#     res_array = InferenceResult[]

#     if !isnothing(group_sizes)
#         ranges_arr = [get_ranges(; group_size = gs, datasize) for gs in group_sizes]
#     elseif !isnothing(group_nbs)
#         ranges_arr = [get_ranges(; group_nb = gn, datasize) for gn in group_numbers]
#     else
#         ArgumentError("Provide whether `group_sizes` or `group_nbs` keyword arguments")
#     end
#     for (i,ranges) in enumerate(ranges_arr)
#         println("***************\nIterative training with $(length(ranges)) segment(s)\n***************")

#         u0s_init = _initialise_u0s_iterative_piecewise_ML(res.pred,res.ranges,ranges)
#         tempres = inference(infprob;
#                                 ranges = ranges, 
#                                 optimizers = optimizers_array[i],
#                                 u0s_init = u0s_init,
#                                 threshold = threshold,
#                                 kwargs...)
#         if tempres.minloss < res.minloss || tempres.minloss < threshold # if threshold is met, we can go one level above
#             push!(res_array, tempres)
#             res = tempres
#         else
#             break
#         end
#     end
#     return res_array
# end

# function _initialise_u0s_iterative_piecewise_ML(pred, ranges_pred, ranges_2)
#     dim_prob = size(first(pred),1)
#     u0_2 = [zeros(eltype(first(pred)), dim_prob) for i in 1:length(ranges_2)]
#     for (i, rng2) in enumerate(ranges_2)
#         _r = first(rng2) # index of new initial condtions on the time steps
#         for j in 0:length(ranges_pred)-1
#             #=
#             NOTE : here we traverse ranges_pred in descending order, to handle overlaps in ranges.
#             Indeed, suppose we go in asending order.
#             if _r == last(rng), it also means that _r == first(next rng),
#             and in this case pred(first(next rng)) estimate is more accurate (all pred in the range depend on its value).
#             =#
#             rng = ranges_pred[end-j]
#             if _r in rng
#                 ui_pred = reshape(pred[end-j][:, _r .== rng],:)
#                 u0_2[i] .= ui_pred
#                 break
#             end
#         end
#     end
#     return u0_2
# end

function _init_u0s(data, ranges)
    u0s_init = [data[:,first(rg)] for rg in ranges]
    return u0s_init
end

function _u0_to_optim_space(u0s_init, u0_bij)
    u0s_init = [u0_bij(u0) for u0 in u0s_init] # projecting u0s_init in optimization space
    return vcat(u0s_init...)
end

function _build_θ(p_init, u0s_init, infprob)
    # initialise p_init
    @unpack u0_bij = infprob
    p̃ = to_optim_space(p_init, infprob)
    ũ0s = _u0_to_optim_space(u0s_init, u0_bij)
    # initialise u0s
    # trainable parameters
    ComponentArray(p̃; u0s = ũ0s)
end