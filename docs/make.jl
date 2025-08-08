using Polfed
using Documenter

DocMeta.setdocmeta!(Polfed, :DocTestSetup, :(using Polfed); recursive=true)

makedocs(;
    modules=[Polfed],
    authors="RockClimbingRocks <rok123.pintar2@gmail.com> and contributors",
    sitename="Polfed.jl",
    format=Documenter.HTML(;
        canonical="https://RockClimbingRocks.github.io/Polfed.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RockClimbingRocks/Polfed.jl",
    devbranch="main",
)
