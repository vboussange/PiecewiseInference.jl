# this file contains utilities for plotting convergence during the training
# trajectory for ensmeble problems, for minibatch_loss simplu put 0
lss = ["-","-."] 
fontdict = Dict("size" => 16)
"""
$(SIGNATURES)

# Arguments:

# Optional
- p_true : optional `Dict("p_true" => [val1, val2], "labs" => ["p1","p2",...] )`
- color_palette : for the time series. For now, the option is not provided to the user.
"""
function plot_convergence(losses, 
                            pred, 
                            data_set,
                            ranges, 
                            tsteps;
                            p_true = nothing, 
                            p_labs = nothing,
                            θs = [],
                            p_trained = [],
                            color_palette = nothing
                            )

    PyPlot.close("all")
    dim_prob = size(pred,1)

    if isnothing(color_palette)
        _cmap = PyPlot.cm.get_cmap("tab20", min(dim_prob,20))
        color_palette = [_cmap(i) for i in 1:min(dim_prob,20)]
    end

    nplots = min(2, size(pred,3)) + 1 + !isnothing(p_true) 
    if nplots == 2 # for EnsembleProblems
        fig, axs = PyPlot.subplots(1, 2, figsize = (10, 5)) # only loss and time series
    elseif nplots == 3
        fig, axs = PyPlot.subplots(1,3, figsize = (15, 5)) # loss, params convergence and time series
    else
        fig, axs = PyPlot.subplots(2,2, figsize = (10, 10)) # loss, params convergence and 2 time series (ensemble problems)
    end

    # plotting loss
    ax = axs[1] 
    ax.plot(1:length(losses), losses, c = "tab:blue", label = "Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontdict=fontdict); ax.set_ylabel("Loss", fontdict=fontdict)

    if !isempty(θs) # plotting parameter convergence
        axx = ax.twinx()
        axx.plot(1:length(θs), θs, c = "tab:red", label = L"|| \hat{\theta} - \theta||")
        axx.set_yscale("log")
        axx.set_xlabel("Iterations",fontdict=fontdict); axx.set_ylabel(L"|| \hat{\theta} - \theta||",fontdict=fontdict)
        axx.legend(fontsize=12)
    end

    # plotting time series
    if pred isa Vector # multiple shooting
        for g in 1:length(ranges)
            _pred = pred[g]
            _tsteps = tsteps[ranges[g]]
            for k in 1:min(2, size(pred,3)) # for EnsembleProblems
                for i in 1:size(_pred,1)
                    axs[k+1].plot(_tsteps, _pred[i,:, k], color = color_palette[i], ls = lss[1], label = (i == 1) && (g == 1) ? "Recovered dynamics" : nothing,)
                end
                for i in 1:size(data_set,1)
                    axs[k+1].plot(_tsteps, data_set[i,ranges[g],k], color = color_palette[i], ls = lss[2], label =  (i == 1) && (g == 1) ? "Data" : nothing)
                end
            end
        end
    else
        for k in 1:min(2, size(pred,3))
            for i in 1:dim_prob # normal run
                axs[k+1].plot(tsteps, pred[i,:,k], color = color_palette[i], ls = lss[1], label = (i == 1) ? "Recovered dynamics" : nothing)
            end
            for i in 1:size(data_set,1)
                axs[k+1].plot(tsteps, data_set[i,:,k], color = color_palette[i], ls = lss[2], label =  (i == 1) ? "Data" : nothing)
            end
        end
    end
    for k in 1:min(2, size(pred,3))
        axs[k+1].set_xlabel("t",fontdict=fontdict); ax.set_ylabel(L"x_\theta(t)",fontdict=fontdict)
    end
    axs[2].legend(fontsize=12)


    # plotting error inferred params
    if !isnothing(p_true) 
        rel_err_p = (p_trained .- p_true) ./ p_true
        ax = axs[end]
        ax.scatter(rel_err_p, 1:length(p_true))
        ax.set_yticks(1:length(p_true))
        !isnothing(p_labs) ? ax.set_yticklabels(p_labs) : nothing
        ax.set_xlabel("Relative error",fontdict=fontdict)
        ax.set_xlim(-0.6,0.6)
        ax.vlines(0., ymin = 0, ymax = length(p_true) + 1, label = "True value", linestyle = "--", color="tab:grey")
        ax.legend(fontsize=12)
    end

    _let = ["A","B","C","D"]
    for (i,ax) in enumerate(axs)
        _x = -0.1
        ax.text(_x, 1.05, _let[i],
            fontsize=16,
            fontweight="bold",
            va="bottom",
            ha="left",
            transform=ax.transAxes ,
        )
    end
    fig.tight_layout()
    # fig.savefig("figs/$name_scenario.pdf", dpi = 1000)
    display(fig)
    return fig
end

# plot_convergence(p_init[dim_prob * trajectories + 1 : end], [p_init[(j-1)*dim_prob+1:j*dim_prob] for j in 1:trajectories], losses, θs, pred)
