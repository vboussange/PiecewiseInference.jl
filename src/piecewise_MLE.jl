# for more intuition on kwargs : https://discourse.julialang.org/t/passing-kwargs-can-overwrite-other-keyword-arguments/74933

"""
$(SIGNATURES)

default loss function for `piecewise_MLE`.
"""
function _loss_multiple_shoot_init(data, params, pred, rg, ic_term)
    l =  mean((data - pred).^2)
    l +=  mean((data[:,1] - pred[:,1]).^2) * ic_term # putting more weights on initial conditions
    return l
end

#=
    Need to overwrite the behavior of length(nc::NCcyle)
    because it does not correspond to what we aim at
=#
import Base.length
import IterTools.NCycle
length(nc::NCycle) = nc.n

"""
$(SIGNATURES)

Piecewise inference. Loops through the optimizers `optimizers`.
Returns a `InferenceResult`.

# Arguments
- `opt` : array of optimizers
- `p_init` : initial guess for parameters of `model`
- `group_size` : size of segments
- `group_nb`: alternatively to `group_size`, one can ask for a certain number of segments
- `data_set` : data
- `model` : a `ParametricModel`, from ParametricModels.jl.
- `tsteps` : corresponding to data

# Optional
- `loss_fn` : the loss function, that takes as arguments `loss_fn(data, params, pred, rg, ic_term)` where 
    `data` is the training data, `params` is the parameter of the model (for defining priors)
    pred` corresponds to the predicted state variables, `rg` corresponds
    to the range of the piecewise wrt the initial data, and `ic_term` is a weight on the initial conditions. 
    `loss_fn` must transform the pred into the observables, with a function 
    `h` that maps the state variables to the observables. By default, `h` is taken as the identity.
- `u0_init` : if not provided, we initialise from `data_set`
- `optimizers` : array of optimizers, e.g. `[Adam(0.01)]`
- `epochs` : number of epochs, which length should match that of `optimizers`
- `batchsizes`: array of batch size, which length should match that of `optimizers`
- `continuity_term` : weight on continuity conditions
- `ic_term` : weight on initial conditions
- `verbose_loss` : displaying loss
- `info_per_its` = 50,
- `plotting` : plotting convergence loss
- `info_per_its` = 50,
- `cb` : call back function.
    Must be of the form `cb(θs, p_trained, losses, pred, ranges)`
- `threshold` : default to 1e-6

# Examples
```julia

using LinearAlgebra, ParametricModels, DiffEqSensitivity
using UnPack
using OptimizationOptimisers, OptimizationFlux
using PiecewiseInference

@model MyModel
function (m::MyModel)(du, u, p, t)
    @unpack b = p
    du .=  0.1 .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])

p_true = (b = [0.23, 0.5],)
p_init= (b = [1., 2.],)

u0 = ones(2)
mp = ModelParams(p_true, 
                tspan,
                u0, 
                BS3(),
                sensealg = ForwardDiffSensitivity();
                saveat = tsteps, 
                )
model = MyModel(mp)
sol_data = simulate(model)
ode_data = Array(sol_data)
optimizers = [ADAM(0.001)]
epochs = [5000]
group_nb = 2
batchsizes = [1]
res = piecewise_MLE(p_init = p_init, 
                    group_nb = group_nb, 
                    data_set = ode_data, 
                    model = model, 
                    tsteps = tsteps, 
                    epochs = epochs, 
                    optimizers = optimizers,
                    batchsizes = batchsizes,
                    )
```
"""
function piecewise_MLE(;group_size = nothing, group_nb = nothing,  kwargs...)
    datasize = size(kwargs[:data_set],2)
    ranges = get_ranges(;group_size, group_nb, datasize)
    _piecewise_MLE(;ranges,  kwargs...)
end

"""
$(SIGNATURES)

Similar to `piecewise_MLE` but for independent time series, where `data_set`
is a vector containing the independent arrays corresponding to the time series,
and `tsteps` is a vector where each entry contains the time steps
of the corresponding time series.
"""
function piecewise_ML_indep_TS(;group_size = nothing, 
                                group_nb = nothing,
                                data_set::Vector, #many different initial conditions
                                tsteps::Vector, #corresponding time steps
                                save_pred = true, # saving prediction
                                kwargs...)
    @assert length(tsteps) == length(data_set) "Independent time series must be gathered as a Vector"
    @assert all(size(data_set[1],1) .== size.(data_set, 1)) "Independent time series must have same state variable dimension"

    datasize_arr = size.(data_set,2)
    ranges_arr = [get_ranges(;group_size, group_nb, datasize = datasize_arr[i]) for i in 1:length(data_set)]
    # updating to take into account the shift provoked by concatenating independent TS
    ranges_shift = cumsum(datasize_arr) # shift
    for i in 2:length(ranges_arr)
        for j in 1:length(ranges_arr[i]) # looping through rng in each independent TS
            ranges_arr[i][j] = ranges_shift[i-1] .+ ranges_arr[i][j] #adding shift to the start of the range
        end
    end
    data_set_cat = cat(data_set...,dims=2)
    ranges_cat = vcat(ranges_arr...)
    tsteps_cat = vcat(tsteps...)

    res = _piecewise_MLE(;ranges=ranges_cat,
                        data_set=data_set_cat, 
                        tsteps=tsteps_cat, 
                        kwargs...,
                        continuity_term = 0.,) 
                        # this overrides kwargs, essential as it does not 
                        # make sense to have continuity across indepdenent TS
                        # NOTE: we could have continuity within a time series, 
                        # this must be carefully thought out.
        
    # reconstructing the problem with original format
    ranges_arr = [get_ranges(;group_size, group_nb, datasize = datasize_arr[i]) for i in 1:length(data_set)]
    idx_res = [0;cumsum(length.(ranges_arr))]

    # group u0s in vector of u0s, 
    # [[u_0_TS1_1, ..., u_0_TS1_n],...,[u_0_TS1_1,...]]
    u0s_trained_arr = [res.u0s_trained[idx_res[i]+1:idx_res[i+1]] for i in 1:length(data_set)]
    
    if save_pred
        # group back the time series in vector, to have
        # pred = [ [mibibatch_1_ts_1, mibibatch_2_ts_1...],  [mibibatch_1_ts_2, mibibatch_2_ts_2...] ...]
        pred_arr = [res.pred[idx_res[i]+1:idx_res[i+1]] for i in 1:length(data_set)]
        res_arr = InferenceResult(res.model,
                            res.minloss,
                            res.p_trained,
                            u0s_trained_arr, 
                            pred_arr, 
                            ranges_arr, 
                            res.losses,)
    else
        res_arr = InferenceResult(res.model,
                            res.minloss,
                            res.p_trained,
                            u0s_trained_arr,
                            [], 
                            ranges_arr, 
                            res.losses,)
    end
    return res_arr
end

function _piecewise_MLE(;p_init, 
                        u0s_init = nothing, # provided by iterative_piecewise_MLE
                        ranges, # provided by piecewise_MLE
                        data_set, 
                        model, 
                        tsteps, 
                        loss_fn = _loss_multiple_shoot_init,
                        optimizers = [ADAM(0.01), BFGS(initial_stepnorm=0.01)],
                        epochs = [1000, 200],
                        batchsizes = fill(length(ranges),length(epochs)),
                        continuity_term = 1.,
                        ic_term = 1.,
                        verbose_loss = true,
                        plotting = false,
                        info_per_its=50,
                        cb = nothing,
                        p_true = nothing,
                        p_labs = nothing,
                        threshold = 1e-16,
                        save_pred = true, 
                        kwargs...
                        )
    dim_prob = get_dims(model) #used by loss_nm
    idx_ranges = (1:length(ranges),) # idx of batches

    @assert (length(optimizers) == length(epochs) == length(batchsizes)) "`optimizers`, `epochs`, `batchsizes` must be of same length"
    @assert (size(data_set,1) == dim_prob) "The dimension of the training data does not correspond to the dimension of the state variables. This probably means that the training data corresponds to observables different from the state variables. In this case, you need to provide manually `u0s_init`." 
    for (i,opt) in enumerate(optimizers)
        OPT = typeof(opt)
        if OPT <: Union{Optim.AbstractOptimizer, Optim.Fminbox, Optim.SAMIN, Optim.ConstrainedOptimizer}
            @assert batchsizes[i] == length(idx_ranges...) "$OPT is not compatible with mini-batches - use `batchsizes = group_nb`"
        end
    end

    # initialise p_init
    p_init = _init_p(p_init, model)
    # initialise u0s
    u0s_init = _init_u0s(u0s_init, data_set, ranges, model)
    # trainable parameters
    θ = [u0s_init;p_init]

    # piecewise loss
    function _loss(θ, idx_rngs)
        return piecewise_loss(θ, 
                            data_set, 
                            tsteps, 
                            model, 
                            (data, params, pred, rg) -> loss_fn(data, params, pred, rg, ic_term),
                            ranges,
                            idx_rngs;
                            continuity_term = continuity_term, 
                            kwargs...)
    end
    __loss(x, p, idx_rngs=idx_ranges...) = _loss(x, idx_rngs) #used for the "Optimization function"

    nb_group = length(ranges)
    println("piecewise_MLE with $(length(tsteps)) points and $nb_group groups.")

    # Here we need a default behavior for Optimization.jl (see https://github.com/SciML/Optimization.jl/blob/c0a51120c7c54a89d091b599df30eb40c4c0952b/lib/OptimizationFlux/src/OptimizationFlux.jl#L53)
    callback(θ, l, pred=[]) = begin
        push!(losses, l)
        p_trained = _get_param(θ,nb_group,dim_prob)
        isnothing(p_true) ? nothing : push!(θs, sum((p_trained .- p_true).^2))
        if length(losses)%info_per_its==0
            verbose_loss ? println("Current loss after $(length(losses)) iterations: $(losses[end])") : nothing
            if !isnothing(cb)
                cb(θs, p_trained, losses, pred, ranges)
            end
            if plotting
                plot_convergence(losses, 
                                pred, 
                                data_set, 
                                ranges, 
                                tsteps, 
                                p_true = p_true, 
                                p_labs = p_labs, 
                                θs = θs, 
                                p_trained = p_trained)
            end
        end
        if l < threshold
            println("❤ Threshold met ❤")
            return true
        else
            return false
        end
    end

    ################
    ### TRAINING ###
    ################
    # Container to track the losses
    losses = Float64[]
    # Container to track the parameter evolutions
    θs = Float64[]


    @info "Training started"
    # TODO: here, not sure whether we use sensealg or not!

    objectivefun = OptimizationFunction(__loss, Optimization.AutoForwardDiff())
    opt = first(optimizers)
    optprob = Optimization.OptimizationProblem(objectivefun, θ)
    res = __solve(opt, optprob, idx_ranges, batchsizes[1], epochs[1], callback)
    for (i, opt) in enumerate(optimizers[2:end])
        optprob = remake(optprob, u0=res.minimizer)
        res = __solve(optimizers[i+1], optprob, idx_ranges, batchsizes[i+1], epochs[i+1], callback)
    end
    
    minloss, pred = _loss(res.minimizer, idx_ranges...)
    p_trained = _get_param(res.minimizer, nb_group, dim_prob) |> collect
    u0s_trained = _get_u0s(res.minimizer, nb_group, dim_prob, model)


    @info "Minimum loss for all batches: $minloss"
    if !isnothing(cb)
        cb(θs, p_trained, losses, pred, ranges)
    end
    if plotting
        plot_convergence(losses, 
                        pred, 
                        data_set, 
                        ranges, 
                        tsteps,
                        θs = θs, 
                        p_trained = p_trained)
    end
    
    if save_pred
        res = InferenceResult(model,
                        minloss, 
                        p_trained,
                        u0s_trained,
                        pred, 
                        ranges, 
                        losses)
    else
        res = InferenceResult(model,
                        minloss,
                        p_trained,
                        u0s_trained, 
                        [], 
                        ranges, 
                        losses,)
    end
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

"""
$(SIGNATURES)

Performs a iterative piecewise MLE, iterating over `group_sizes`. 
Stops the iteration when loss function increases between two iterations.

Returns an array with all `InferenceResult` obtained during the iteration.
For kwargs, see `piecewise_MLE`.

# Note 
- for now, does not support independent time series (`piecewise_ML_indep_TS`).
- at every iteration, initial conditions are initialised given the predition of previous iterations

# Specific arguments
- `group_sizes` : array of group sizes to test
- `optimizers_array`: optimizers_array[i] is an array of optimizers for the trainging processe of `group_sizes[i`
"""
function iterative_piecewise_MLE(;group_sizes = nothing,
                                group_nbs = nothing,
                                optimizers_array,
                                threshold = 1e-16,
                                kwargs...)

    @assert length(group_sizes) == length(optimizers_array)

    # initialising results
    data_set = kwargs[:data_set]
    datasize = size(data_set,2)
    model = kwargs[:model]
    p_trained, _ = destructure(kwargs[:p_init])
    res = InferenceResult(model = model,
                        p_trained = p_trained,
                        pred = [data_set], 
                        ranges = [1:datasize])
    res_array = InferenceResult[]

    if !isnothing(group_sizes)
        ranges_arr = [get_ranges(; group_size = gs, datasize) for gs in group_sizes]
    elseif !isnothing(group_nbs)
        ranges_arr = [get_ranges(; group_nb = gn, datasize) for gn in group_numbers]
    else
        ArgumentError("Provide whether `group_sizes` or `group_nbs` keyword arguments")
    end
    for (i,ranges) in enumerate(ranges_arr)
        println("***************\nIterative training with $(length(ranges)) segment(s)\n***************")

        u0s_init = _initialise_u0s_iterative_piecewise_ML(res.pred,res.ranges,ranges)
        tempres = _piecewise_MLE(;ranges = ranges, 
                                optimizers = optimizers_array[i],
                                u0s_init = u0s_init,
                                threshold = threshold,
                                kwargs...)
        if tempres.minloss < res.minloss || tempres.minloss < threshold # if threshold is met, we can go one level above
            push!(res_array, tempres)
            res = tempres
        else
            break
        end
    end
    return res_array
end

function _initialise_u0s_iterative_piecewise_ML(pred, ranges_pred, ranges_2)
    dim_prob = size(first(pred),1)
    u0_2 = [zeros(eltype(first(pred)), dim_prob) for i in 1:length(ranges_2)]
    for (i, rng2) in enumerate(ranges_2)
        _r = first(rng2) # index of new initial condtions on the time steps
        for j in 0:length(ranges_pred)-1
            #=
            NOTE : here we traverse ranges_pred in descending order, to handle overlaps in ranges.
            Indeed, suppose we go in asending order.
            if _r == last(rng), it also means that _r == first(next rng),
            and in this case pred(first(next rng)) estimate is more accurate (all pred in the range depend on its value).
            =#
            rng = ranges_pred[end-j]
            if _r in rng
                ui_pred = reshape(pred[end-j][:, _r .== rng],:)
                u0_2[i] .= ui_pred
                break
            end
        end
    end
    return u0_2
end

function __solve(opt::OPT, optprob, idx_ranges, batchsizes, epochs, callback) where OPT <: Union{Optim.AbstractOptimizer, 
                                                                                                Optim.Fminbox,
                                                                                                Optim.SAMIN, 
                                                                                                Optim.ConstrainedOptimizer}
    @info "Running optimizer $OPT"
    res = Optimization.solve(optprob,
                            opt,
                            maxiters = epochs, 
                            callback = callback)
    return res
end

function __solve(opt::OPT, optprob, idx_ranges, batchsizes, epochs, callback) where OPT
    @info "Running optimizer $OPT"
    train_loader = Flux.Data.DataLoader(idx_ranges; batchsize = batchsizes, shuffle = true, partial=false)
    res = Optimization.solve(optprob,
                            opt, 
                            ncycle(train_loader, epochs),
                            callback=callback, 
                            save_best=true)
    return res
end

function _init_p(p_init, model)
    p_init, _ = Optimisers.destructure(p_init)
    p_init = get_p_bijector(model)(p_init) # projecting p_init in optimization space
    return p_init
end

"""
    $SIGNATURES
`u0s_init` should come as a vector of u0, i.e. a vector of vector.
We ouput it as a vector of scalar, after tranformation in the optimization space.
"""
function _init_u0s(u0s_init, data_set, ranges, model)
     # initialising with data_set if not provided
     if isnothing(u0s_init) 
        u0s_init = [data_set[:,first(rg)] for rg in ranges]
    end
    u0s_init = [get_u0_bijector(model)(u0) for u0 in u0s_init] # projecting u0s_init in optimization space
    return vcat(u0s_init...)
end