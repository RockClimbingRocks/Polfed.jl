
using Documenter, Literate, Polfed

# Convert Literate scripts into Markdown
# Literate.markdown("/home/rokpintar/projects/Polfed/examples/mwe.jl", "docs/src/examples")
# Literate.markdown("examples/example2.jl", "docs/src/examples")

Literate.markdown("/home/rokpintar/projects/Polfed/docs/src/documentation/docs-polfed.jl", "src/documentation/generated", execute=false)


makedocs(
    modules=[Polfed],
    authors="RockClimbingRocks <rok123.pintar2@gmail.com> and contributors",
    sitename="Polfed.jl",
    format=Documenter.HTML(;
        canonical="https://RockClimbingRocks.github.io/Polfed.jl",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",

        "Documentation" => [
            "Polfed docs" => "documentation/generated/docs-polfed.md",
            # "Lanczos factorization" => "documentation/factorization.md",
            # "Spectral Transformation" => "documentation/spectral_transformation.md",
        ],

        # "Quick Start" => [
        #     "Something1" => "quickstart/something1.md",
        #     "Something2" => "quickstart/something2.md",
        # ],

        # "Examples" => [
        #     "Minimal Working Example" => "examples/mwe.md",
        # ],
    ],
)

deploydocs(
    repo = "github.com/RockClimbingRocks/Polfed.jl.git",
    devbranch="main",
)