using Documenter
using PiecewiseInference

makedocs(
    sitename = "PiecewiseInference",
    format = Documenter.HTML(),
    modules = [PiecewiseInference]
)

deploydocs(repo = "github.com/vboussange/PiecewiseInference.jl", devbranch="main")

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
