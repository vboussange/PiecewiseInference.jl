
@group
    group_size_init = 11
    datasize = 100
    ranges_init = DiffEqFlux.group_ranges(datasize, group_size_init)
    group_size_2 = 21
    ranges_2 = DiffEqFlux.group_ranges(datasize, group_size_2)
    pred_init = [cumsum(ones(3, length(rng)), dims=2) for rng in ranges_init]

    u0_2 = _initialise_u0s_iterative_minibatch_ML(pred_init, ranges_init, ranges_2)
    @test all(u0_2 .== 1.)