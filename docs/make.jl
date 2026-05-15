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
        canonical="https://RockClimbingRocks.github.io/Polfed.jl",
        edit_link=nothing,
        assets=String[],
    ),
    pages=[
        hide("index.md"),
        "Citation" => "citation/index.md",

        "Getting Started" => "getting-started/index.md",

        "Tutorials" => [
            "Beginner" => [
                "Choosing Target" => "tutorials/choosing-target/index.md",
                "Reporting" => "tutorials/reporting/index.md",
                "Lanczos and Block Lanczos Factorization" => "tutorials/lanczos-block-lanczos/index.md",
                "Parallelization" => "tutorials/parallelization/index.md",
                "Optimized Mapping" => "tutorials/optimized-mapping/index.md",
                "Reducing Memory Access" => "tutorials/reducing-memory-access/index.md",
                "Hermitian matrices" => "tutorials/hermitian-matrices/index.md",
                "Working with GPUs" => "tutorials/working-with-gpus/index.md",
            ],
            "Advanced" => [
                "Optimization of the XXZ Model" => "tutorials/optimization-of-the-xxz-model/index.md",
                "Custom Mapping" => "tutorials/custom-mapping/index.md",
                "Automatic Optimization" => "tutorials/automatic-optimization/index.md",
                "GPU Implementation" => "tutorials/gpu-implementation/index.md",
            ],
        ],

        "Models" => [
            "Quantum Sun (QSun)" => "models/qsun/index.md",
            "XXZ" => "models/xxz/index.md",
            "J1-J2" => "models/j1j2/index.md",
        ],

        "Guidelines" => "guidelines/index.md",

        "Documentation" => [
            "Core Functions" => "documentation/core-functions/index.md",
            "Models" => "documentation/models/index.md",
            "Configs and Parallelization Types" => "documentation/configs-and-parallelization-types/index.md",
            "Reports, Logging, and Defaults" => "documentation/reports-logging-and-defaults/index.md",
        ],
        "FAQ / Troubleshooting" => "faq/index.md",
    ],
)

deploydocs(
    repo="github.com/RockClimbingRocks/Polfed.jl.git",
    devbranch="dev",
)
