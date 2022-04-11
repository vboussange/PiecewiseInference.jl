# this file contains utilities for plotting convergence during the training
# trajectory for ensmeble problems, for minibatch_loss simplu put 0
lss = ["-","-."] 
fontdict = Dict("size" => 16)
"""
    plot_convergence(p_trained, 
                    losses, 
                    pred, 
                    ranges, 
                    tsteps;
                    p_true_dict = nothing, 
                    θs = []
                    )
               
# Arguments:

# Optional
- p_true : optional `Dict("p_true" => [val1, val2], "labs" => ["p1","p2",...] )`
- 
"""
function plot_convergence(losses, 
                            pred, 
                            data_set,
                            ranges, 
                            tsteps;
                            p_true_dict, 
                            θs,
                            p_trained
                            )

    close("all")
    dim_prob = size(data_set,1)
    nplots = min(2, size(pred,3)) + 1 + !isempty(p_trained)
    if nplots == 2 # for EnsembleProblems
        fig, axs = subplots(1, 2, figsize = (10, 5)) # only loss and time series
    elseif nplots == 3
        fig, axs = subplots(1,3, figsize = (15, 5)) # loss, params convergence and time series
    else
        fig, axs = subplots(2,2, figsize = (10, 10)) # loss, params convergence and 2 time series (ensemble problems)
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
            for i in 1:dim_prob
                for k in 1:min(2, size(pred,3)) # for EnsembleProblems
                    axs[k+1].plot(_tsteps, _pred[i,:, k], color = color_palette[i], ls = lss[1], label = (i == 1) && (g == 1) ? "Recovered dynamics" : nothing,)
                    axs[k+1].plot(_tsteps, data_set[i,ranges[g],k], color = color_palette[i], ls = lss[2], label =  (i == 1) && (g == 1) ? "Data" : nothing)
                end
            end
        end
    else
        for i in 1:dim_prob # normal run
            for k in 1:min(2, size(pred,3))
                axs[k+1].plot(tsteps, pred[i,:,k], color = color_palette[i], ls = lss[1], label = (i == 1) ? "Recovered dynamics" : nothing)
                axs[k+1].plot(tsteps, data_set[i,:,k], color = color_palette[i], ls = lss[2], label =  (i == 1) ? "Data" : nothing)
            end
        end
    end
    for ax in axs[2:end]
        ax.set_xlabel("t",fontdict=fontdict); ax.set_ylabel(L"x_\theta(t)",fontdict=fontdict)
    end
    axs[2].legend(fontsize=12)


    # plotting error inferred params
    if !isnothing(p_true_dict) 
        @unpack p_true, lab = p_true_dict
        rel_err_p = (p_trained .- p_true) ./ p_true
        ax = axs[end]
        ax.scatter(rel_err_p, 1:length(p_true))
        ax.set_yticks(1:length(lab))
        ax.set_yticklabels(lab)
        ax.set_xlabel("Relative error",fontdict=fontdict)
        ax.set_xlim(-0.6,0.6)
        ax.vlines(0., ymin = 0, ymax = length(lab) + 1, label = "True value", linestyle = "--", color="tab:grey")
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
