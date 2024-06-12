# PiecewiseInference.jl
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vboussange.github.io/PiecewiseInference.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/PiecewiseInference.jl/dev/)
[![Build status (Github Actions)](https://github.com/vboussange/PiecewiseInference.jl/workflows/CI/badge.svg)](https://github.com/vboussange/PiecewiseInference.jl/actions)
[![codecov.io](http://codecov.io/github/vboussange/PiecewiseInference.jl/coverage.svg?branch=main)](http://codecov.io/github/vboussange/PiecewiseInference.jl?branch=main)

**PiecewiseInference.jl** is a library to enhance the convergence of dynamical model parameter inversion methods. It provides features such as
- a segmentation strategy, 
- the independent estimation of initial conditions for each segment, 
- parameter transformation, 
- parameter and initial conditions regularization
- mini-batching

Taken altogether, these features regularize the inference problem and permit to solve it efficiently.

![](docs/animated.gif)

## Installation
Open Julia REPL and type
```julia
using Pkg; Pkg.add(url="https://github.com/vboussange/PiecewiseInference.jl")
```

That's it! This will download the latest version of **PiecewiseInference.jl** from this git repo and download all dependencies.


## Getting started

Check out [this blog post](https://vboussange.github.io/post/piecewiseinference/) providing a hands-on tutorial.
See also the API documentation and the `test` folder.

## Related packages
`DiffEqFlux` is a package with similar goals as `PiecewiseInference`, and proposes the method `DiffEqFlux.multiple_shooting`, which is close to `PiecewiseInference.inference` but where initial conditions are not inferred. `PiecewiseInference` further proposes several utility methods for model selection.

## Reference
Boussange, V., Vilimelis-Aceituno, P., Schäfer, F., Pellissier, L., _Partitioning time series to improve process-based models with machine learning_. [[bioRxiv]](https://www.biorxiv.org/content/10.1101/2022.07.25.501365v2) (2024), 46 pages.
