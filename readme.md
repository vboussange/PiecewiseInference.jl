# PiecewiseInference.jl
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vboussange.github.io/PiecewiseInference.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/PiecewiseInference.jl/dev/)
[![Build status (Github Actions)](https://github.com/vboussange/PiecewiseInference.jl/workflows/CI/badge.svg)](https://github.com/vboussange/PiecewiseInference.jl/actions)
[![codecov.io](http://codecov.io/github/vboussange/PiecewiseInference.jl/coverage.svg?branch=main)](http://codecov.io/github/vboussange/PiecewiseInference.jl?branch=main)

Suite for inverse modelling of dynamical systems characterised by complex dynamics. 

**PiecewiseInference.jl** implements a **segmentation method** that improves the convergence of local search methods by regularizing the inference problem.

![](docs/animated.gif)

## Installation
PiecewiseInference.jl has [ParametricModels.jl](https://github.com/vboussange/ParametricModels.jl) in its dependency, a non-registered package. As such, to install PiecewiseInference.jl, you'll need to first add an extra registry to your Julia installation that tracks both ParametricModels.jl and PiecewiseInference.jl.

To proceed, open Julia and type the following
```julia
using Pkg
pkg"registry add https://github.com/vboussange/VBoussangeRegistry.git"
```
Then go on and 
```julia
pkg"add PiecewiseInference"
```

That's it! This will download the latest version of **PiecewiseInference.jl** from this git repo and download all dependencies.


## Getting started

Check out [this blog post](https://vboussange.github.io/post/piecewiseinference/) providing a hands-on tutorial.
See also the documentation and the `test` folder.

## Related packages
`DiffEqFlux` is a package with similar goals as `PiecewiseInference`, and proposes the method `DiffEqFlux.multiple_shooting`, which is close to `PiecewiseInference.inference` but where initial conditions are not inferred. `PiecewiseInference` further proposes several utility methods for model selection.

## Reference
- Boussange, V., Vilimelis-Aceituno, P., Pellissier, L., _Mini-batching ecological data to improve ecosystem models with machine learning_ [[bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.25.501365v1)] (2022), 46 pages.
