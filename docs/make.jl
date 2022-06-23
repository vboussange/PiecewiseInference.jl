using Documenter
using EcologyInformedML

makedocs(
    sitename = "EcologyInformedML",
    format = Documenter.HTML(),
    modules = [EcologyInformedML]
)

deploydocs(repo = "github.com/vboussange/EcologyInformedML.jl", devbranch="main")

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
