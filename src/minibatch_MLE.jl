"""
    _loss_multiple_shoot_init(data, pred, ic_term)
default loss function for `minibatch_MLE`.
"""
function _loss_multiple_shoot_init(data, pred, ic_term)
    l =  mean((data - pred).^2)
    l +=  mean((data[:,1] - pred[:,1]).^2) * ic_term # putting more weights on initial conditions
    return l
end

"""
    minibatch_MLE(;p_init, 
                group_size, 
                data_set, 
                prob, 
                tsteps, 
                alg, 
                sensealg,
                loss_fn = _loss_multiple_shoot_init(data, pred, ic_term),
                λ = Dict("ADAM" => 0.01, "BFGS" => 0.01),
                maxiters = Dict("ADAM" => 2000, "BFGS" => 1000),
                continuity_term = 1.,
                ic_term = 1.,
                verbose = true,
                plotting = false,
                p_true = nothing,
                p_labs = nothing,
                threshold = 1e-6)

Maximum likelihood estimation with minibatching. Loops through ADAM and BFGS.
Returns `minloss, p_trained, ranges, losses, θs`.

# arguments
- p_init : initial guess for parameters of `prob`
- group_size : size of segments
- data_set : data
- prob : ode problem
- tsteps : corresponding to data
- alg : ODE solver
- sensealg : sensitivity solver

# optional
- u0_init : if not provided, we initialise from `data_set`
- `loss_fn` : loss function with arguments `loss_fn(data, pred, ic_term)`
- `λ` : dictionary with learning rates. `Dict("ADAM" => 0.01, "BFGS" => 0.01)`
- `maxiters` : dictionary with maximum iterations. Dict("ADAM" => 2000, "BFGS" => 1000),
- `continuity_term` : weight on continuity conditions
- `ic_term` : weight on initial conditions
- `verbose` : displaying loss
- `plotting` : plotting convergence loss
- `saving_plots` = false : saves plotting figure in dir `saving_dir`
- `saving_dir` = "plot_convergence" : directory and name without extension of the plotting file
- `p_true` : true params
- `p_labs` : labels of the true parameters
- `threshold` : default to 1e-6
"""
function minibatch_MLE(;group_size,  kwargs...)
    datasize = size(kwargs[:data_set],2)
    println("hey")
    _minibatch_MLE(;ranges=_get_ranges(group_size, datasize),  kwargs...)
end

function _minibatch_MLE(;p_init, 
                        u0s_init = nothing,
                        ranges, 
                        data_set, 
                        prob, 
                        tsteps, 
                        alg, 
                        sensealg,
                        loss_fn = _loss_multiple_shoot_init,
                        λ = Dict("ADAM" => 0.01, "BFGS" => 0.01),
                        maxiters = Dict("ADAM" => 2000, "BFGS" => 1000),
                        continuity_term = 1.,
                        ic_term = 1.,
                        verbose = true,
                        plotting = false,
                        saving_plots = false,
                        saving_dir = "plot_convergence",
                        info_per_its=50,
                        p_true = nothing,
                        p_labs = nothing,
                        threshold = 1e-16,
                        )
    dim_prob = length(prob.u0) #used by loss_nm

    # minibatch loss
    function loss_mb(θ)
        return minibatch_loss(θ, 
                            data_set, 
                            tsteps, 
                            prob, 
                            (data, pred) -> loss_fn(data, pred, ic_term),
                            alg, 
                            ranges, 
                            continuity_term = continuity_term, 
                            senselag = sensealg)
    end

    # normal loss
    function loss_nm(θ)
        params = @view θ[dim_prob + 1: end] # params of the problem
        u0_i = @view θ[1:dim_prob]
        prob_i = remake(prob; p=params, tspan=(tsteps[1], tsteps[end]), u0=u0_i)
        sol = solve(prob_i, alg; saveat=tsteps, senselag = sensealg)
        sol.retcode == :Success ? nothing : return Inf, []
        pred = sol |> Array
        l = loss_fn(data_set, pred, ic_term)
        return l, pred
    end

    if length(ranges) > 1
        # minibatching
        _loss = loss_mb
    else
        # normal MLE with initial estimation
        _loss = loss_nm
    end

    # initialising with data_set if not provided
    isnothing(u0s_init) ? u0s_init = reshape(data_set[:,first.(ranges),:],:) : nothing
    # making sure that u0s_init are positive, otherwise we might have some numerical difficulties
    u0s_init[u0s_init .< 0.] .= 1e-3
    θ = [u0s_init;p_init]
    nb_group = length(ranges)
    println("minibatch_MLE with $(length(tsteps)) points and $nb_group groups.")

    callback(θ, l, pred) = begin
        push!(losses, l)
        p_trained = @view θ[nb_group * dim_prob + 1: end]
        isnothing(p_true) ? nothing : push!(θs, sum((p_trained .- p_true).^2))
        if length(losses)%info_per_its==0
            verbose ? println("Current loss after $(length(losses)) iterations: $(losses[end])") : nothing
            if plotting
                fig = plot_convergence(losses, 
                                        pred, 
                                        data_set, 
                                        ranges, 
                                        tsteps, 
                                        p_true = p_true, 
                                        p_labs = p_labs, 
                                        θs = θs, 
                                        p_trained = p_trained)
                saving_plots ? fig.savefig(saving_dir*"-$(length(losses)÷info_per_its).png", dpi = 500) : nothing
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


    println("***************\nTraining started\n***************")
    println("`ADAM` with λ = $(λ["ADAM"])")
    res3 = DiffEqFlux.sciml_train(_loss, 
                                θ, 
                                ADAM(λ["ADAM"]), 
                                cb=callback, 
                                maxiters = maxiters["ADAM"])

    println("`BFGS` with λ = $(λ["BFGS"])")
    res = DiffEqFlux.sciml_train(_loss, 
                                res3.minimizer, 
                                BFGS(initial_stepnorm = λ["BFGS"]), 
                                cb= callback, 
                                maxiters = maxiters["BFGS"])

    minloss, pred = _loss(res.minimizer)
    p_trained = res.minimizer[dim_prob * nb_group + 1 : end]
    return ResultMLE(minloss, p_trained, p_true, p_labs, pred, ranges, losses, θs)
end

function _get_ranges(group_size, datasize)
    if group_size-1 < datasize
        ranges = DiffEqFlux.group_ranges(datasize, group_size)
        # minibatching
    else
        ranges = [1:datasize]
        # normal MLE with initial estimation
    end
    return ranges
end


"""
    iterative_minibatch_MLE(; 
                    group_sizes,
                    learning_rates,
                    kwargs...)
Performs a iterative minibatch MLE, iterating over `group_sizes`. For kwargs, see `minibatch_MLE`.

# arguments
- `group_sizes` : array of group sizes to test
- `learning_rates`: array of dictionary with learning rates for ADAM and BFGS
"""
function iterative_minibatch_MLE(;group_sizes,
                                learning_rates,
                                kwargs...)

    @assert length(group_sizes) == length(learning_rates)

    # initialising results
    data_set = kwargs[:data_set]
    datasize = size(data_set,2)
    res = ResultMLE(Inf, [], [], [], [data_set], [1:datasize], [], [])
    for (i,gs) in enumerate(group_sizes)
        println("***************\nIterative training with group size $gs\n***************")
        ranges = _get_ranges(group_sizes[i], datasize)
        u0s_init = _initialise_u0s_iterative_minibatch_ML(res.pred,res.ranges,ranges)
        tempres = _minibatch_MLE(;ranges = ranges, 
                                λ = learning_rates[i],
                                u0s_init = reshape(u0s_init,:),
                                kwargs...)
        if tempres.minloss < res.minloss || tempres.minloss < kwargs[:threshold] # if threshold is met, we can go one level above
            res = tempres
        else
            break
        end
    end
    return res
end

function _initialise_u0s_iterative_minibatch_ML(pred, ranges_pred, ranges_2)
    dim_prob = size(first(pred),1)
    u0_2 = zeros(eltype(first(pred)), dim_prob, length(ranges_2))
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
                u0_2[:,i] .= ui_pred
                break
            end
        end
    end
    return u0_2
end