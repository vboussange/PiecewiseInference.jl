
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
                p_true_dict = nothing,)

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
- loss_fn : loss function with arguments `loss_fn(data, pred, ic_term)`
- λ : dictionary with learning rates. `Dict("ADAM" => 0.01, "BFGS" => 0.01)`
- maxiters : dictionary with maximum iterations. Dict("ADAM" => 2000, "BFGS" => 1000),
- continuity_term : weight on continuity conditions
- ic_term : weight on initial conditions
- verbose : displaying loss
- plotting : plotting convergence loss
- p_true_dict : nothing
- threshold = 1e-6
"""
function minibatch_MLE(;p_init, 
                        group_size, 
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
                        p_true_dict = nothing,
                        threshold = 1e-6
                        )
    datasize = size(data_set,2)
    dim_prob = length(prob.u0) #used by loss_nm

    @assert mod(datasize,(group_size-1)) == 0. "`group_size` is not compatible with `datasize`\n`group_size` must be a divisor of `size(data_set,2)`"

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

    if group_size-1 < datasize
        ranges = DiffEqFlux.group_ranges(datasize, group_size)
        # minibatching
        _loss = loss_mb
    else
        ranges = [1:datasize]
        # normal MLE with initial estimation
        _loss = loss_nm
    end
    u0s_init = data_set[:,first.(ranges),:][:]
    # making sure that u0s_init are positive, otherwise we might have some numerical difficulties
    u0s_init[u0s_init .< 0.] .= 1e-3
    p_init = [u0s_init;p_init]
    nb_group = length(ranges)
    println("minibatch_MLE with $(length(tsteps)) points and $nb_group groups.")

    callback(θ, l, pred) = begin
        push!(losses, l)
        p_trained = @view θ[nb_group * dim_prob + 1: end]
        isnothing(p_true_dict) ? nothing : push!(θs, sum((p_trained .- p_true_dict["p_true"]).^2))
        if length(losses)%50==0
            verbose ? println("Current loss after $(length(losses)) iterations: $(losses[end])") : nothing
            if plotting
                plot_convergence(losses, pred, data_set, ranges, tsteps, p_true_dict = p_true_dict, θs = θs, p_trained = p_trained)
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
                                p_init, 
                                ADAM(λ["ADAM"]), 
                                cb=callback, 
                                maxiters = maxiters["ADAM"])

    println("`BFGS` with λ = $(λ["BFGS"])")
    res = DiffEqFlux.sciml_train(_loss, 
                                res3.minimizer, 
                                BFGS(initial_stepnorm = λ["BFGS"]), 
                                cb= callback, 
                                maxiters = maxiters["BFGS"])


    p_trained = res.minimizer[dim_prob * nb_group + 1 : end]
    minloss = res.minimum
    println("Final training loss after $(length(losses)) iterations $(losses[end])")
    println("Parameters p = ", p_trained)
    isnothing(p_true_dict) ? nothing : println("True parameters = ", p_true )
    return minloss, p_trained, ranges, losses, θs
end

"""
    recursive_minibatch_MLE(; 
                    group_sizes,
                    learning_rates,
                    kwargs...)

# arguments
- `group_sizes` : array of group sizes to test
- `learning_rates`: array of dictionary with learning rates for ADAM and BFGS
"""
function recursive_minibatch_MLE(;group_sizes,
                                learning_rates,
                                kwargs...)

    @assert length(group_sizes) == length(learning_rates)
    minloss = Inf
    p_trained = kwargs[:p_init]
    ranges = []
    losses = eltype(p_trained)[]
    θs = eltype(p_trained)[]
    for (i,gs) in enumerate(group_sizes)
        println("***************\nRecursive training with group size $gs\n***************")
        temp = minibatch_MLE(;group_size = group_sizes[i], λ = learning_rates[i], kwargs...)
        if temp[1] < minloss
            minloss, p_trained, ranges, losses, θs = temp
        else
            break
        end
    end
    return minloss, p_trained, ranges, losses, θs
end