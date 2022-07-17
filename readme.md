# MiniBatchInference
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vboussange.github.io/MiniBatchInference.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/MiniBatchInference.jl/dev/)
[![Build status (Github Actions)](https://github.com/vboussange/MiniBatchInference.jl/workflows/CI/badge.svg)](https://github.com/vboussange/MiniBatchInference.jl/actions)
[![codecov.io](http://codecov.io/github/vboussange/MiniBatchInference.jl/coverage.svg?branch=main)](http://codecov.io/github/vboussange/MiniBatchInference.jl?branch=main)

Suite for parameter inference and model selection with dynamical models characterised by complex dynamics. `MiniBatchInference` proposes **mini-batching methods** which split observation data in small batches to **regularise the inference process**.

## Installation
Open Julia and type the following
```julia
using Pkg
Pkg.add("https://github.com/vboussange/MiniBatchInference.jl")
```
This will download the latest version from the git repo and download all dependencies.


## Getting started
See the documentation and the `test` folder for up-to-date examples.

## Related packages
`DiffEqFlux` is a package with similar goals as `MiniBatchInference`, and proposes the method `DiffEqFlux.multiple_shooting`, which is close to `MiniBatchInference.minibatch_MLE` but where initial conditions are not inferred. `MiniBatchInference` further proposes several utility methods for model selection, and aims in a near future at proposing full bayesian inference, bridging with `Turing`.