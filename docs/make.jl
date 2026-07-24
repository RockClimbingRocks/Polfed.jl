using UniPolfed
using Documenter

DocMeta.setdocmeta!(UniPolfed, :DocTestSetup, :(using UniPolfed); recursive=true)

makedocs(;
    modules=[UniPolfed],
    authors="RockClimbingRocks <rok123.pintar2@gmail.com> and contributors",
    sitename="UniPolfed.jl",
    format=Documenter.HTML(;
        canonical="https://RockClimbingRocks.github.io/UniPolfed.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RockClimbingRocks/UniPolfed.jl",
    devbranch="main",
)
