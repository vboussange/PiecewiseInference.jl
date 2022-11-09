using PiecewiseInference
using Test

datasize = 205
group_nb = 21
group_size = 10+1

@test all(get_ranges(;datasize, group_size) .== get_ranges(;datasize, group_nb))
@test get_ranges(;datasize, group_size = 205) == [1:datasize]
@test get_ranges(;datasize, group_nb = 1) == [1:datasize]