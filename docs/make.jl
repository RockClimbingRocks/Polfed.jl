using Documenter, Polfed

makedocs(
    modules=[Polfed],
    authors="RockClimbingRocks <rok123.pintar2@gmail.com> and contributors",
    sitename="Polfed.jl",
    remotes=nothing,
    checkdocs=:none,
    doctest=false,
    pagesonly=true,
    format=Documenter.HTML(
        canonical="https://rockclimbingrocks.github.io/Polfed.jl/",
        repolink="https://github.com/RockClimbingRocks/Polfed.jl",
        edit_link=nothing,
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",

        "Getting Started" => [
            "Installation" => "getting-started/installation/index.md",
            "Quickstart" => "getting-started/quickstart/index.md",
        ],

        "How POLFED works" => [
            "The workflow" => "how-polfed-works/index.md",
            "Krylov background" => "how-polfed-works/krylov/index.md",
            "Filters" => "how-polfed-works/filters/index.md",
            "On-the-fly mapping" => "how-polfed-works/on-the-fly-mapping/index.md",
        ],

        "Tutorials" => [
            "POLFED" => "tutorials/polfed/index.md",
            "Double POLFED" => "tutorials/double-polfed/index.md",
            "Patched POLFED" => "tutorials/patched-polfed/index.md",
            "Full POLFED" => "tutorials/full-polfed/index.md",
            "Unitary POLFED" => "tutorials/unitary-polfed/index.md",
        ],

        "Going faster" => [
            "The bottleneck" => "going-faster/index.md",
            "Optimized kernels" => "going-faster/optimized-kernels/index.md",
            "Parallelization" => "going-faster/parallelization/index.md",
            "GPU" => "going-faster/gpu/index.md",
        ],

        "Models" => [
            "Quantum Sun (QSun)" => "models/qsun/index.md",
            "XXZ" => "models/xxz/index.md",
            "J1-J2" => "models/j1j2/index.md",
        ],

        "Guidelines" => "guidelines/index.md",

        "Benchmarks" => "benchmarks/index.md",

        "API Reference" => [
            "Entrypoints" => "api/entrypoints/index.md",
            "Configs and Targets" => "api/configs-and-targets/index.md",
            "Results, Reports, Logging" => "api/results-reports-logging/index.md",
            "Models" => "api/models/index.md",
        ],

        "FAQ / Troubleshooting" => "faq/index.md",

        "Development" => [
            "Architecture" => "development/architecture.md",
            "Roadmap" => "development/roadmap.md",
        ],

        "Citation" => "citation/index.md",
    ],
)

if get(ENV, "DOCUMENTER_DEPLOY", "false") == "true"
    deploydocs(
        repo="github.com/RockClimbingRocks/Polfed.jl.git",
        devbranch="main",
    )
end
