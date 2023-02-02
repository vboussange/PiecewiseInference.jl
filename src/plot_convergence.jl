# this file contains utilities for plotting convergence during the training
# trajectory for ensmeble problems, for piecewise_loss simplu put 0
lss = ["-","-."] 
fontdict = Dict("size" => 16)
"""
$(SIGNATURES)

Returns `fig, axs`

# Optional
- `p_true` : optional `Dict("p_true" => [val1, val2], "labs" => ["p1","p2",...] )`
- `color_palette` : Palette of colors. Must be of size `dim_prob`.
"""
function plot_convergence(losses, 
                            pred, 
                            data_set,
                            ranges, 
                            tsteps)

    PyPlot.close("all")
    dim_prob = size(data_set,1)

    if isnothing(color_palette)
        _cmap = PyPlot.cm.get_cmap("tab20", min(dim_prob,20))
        color_palette = [_cmap(i) for i in 1:min(dim_prob,20)]
    end

    fig, axs = PyPlot.subplots(1, 2, figsize = (10, 5)) # only loss and time series
 

    # plotting loss
    ax = axs[1] 
    ax.plot(1:length(losses), losses, c = "tab:blue", label = "Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontdict=fontdict); ax.set_ylabel("Loss", fontdict=fontdict)

    # plotting time series
    if pred isa Vector # multiple shooting
        for g in 1:length(ranges)
            _pred = pred[g]
            _tsteps = tsteps[ranges[g]]
            for i in 1:size(_pred,1)
                axs[2].plot(_tsteps, _pred[i,:, k], color = color_palette[i], ls = lss[1], label = (i == 1) && (g == 1) ? "Recovered dynamics" : nothing,)
            end
            for i in 1:size(data_set,1)
                axs[2].scatter(_tsteps, data_set[i,ranges[g],k], color = color_palette[i], ls = lss[2], label =  (i == 1) && (g == 1) ? "Data" : nothing)
            end
        end
    else
        for i in 1:dim_prob # normal run
            axs[2].plot(tsteps, pred[i,:,k], color = color_palette[i], ls = lss[1], label = (i == 1) ? "Recovered dynamics" : nothing)
        end
        for i in 1:size(data_set,1)
            axs[2].scatter(tsteps, data_set[i,:,k], color = color_palette[i], ls = lss[2], label =  (i == 1) ? "Data" : nothing)
        end
    end

    axs[2].set_xlabel("t",fontdict=fontdict); ax.set_ylabel(L"x_\theta(t)",fontdict=fontdict)
    axs[2].legend(fontsize=12)

    fig.tight_layout()
    # fig.savefig("figs/$name_scenario.pdf", dpi = 1000)
    display(fig)
    return fig, axs
end

# plot_convergence(p_init[dim_prob * trajectories + 1 : end], [p_init[(j-1)*dim_prob+1:j*dim_prob] for j in 1:trajectories], losses, θs, pred)
