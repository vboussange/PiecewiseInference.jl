using Documenter
using MiniBatchInference

makedocs(
    sitename = "MiniBatchInference",
    format = Documenter.HTML(),
    modules = [MiniBatchInference]
)

deploydocs(repo = "github.com/vboussange/MiniBatchInference.jl", devbranch="main")

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
