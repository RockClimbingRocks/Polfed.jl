using Documenter, Literate, Polfed

files_to_literate = [
    "src/documentation/0.Home.jl",
    "src/documentation/1.Getting_Started.jl",

    # Beginner tutorials
    "src/documentation/2.Tutorial-2.Choosing_Target.jl",
    "src/documentation/2.Tutorial-8.Reporting.jl",
    "src/documentation/2.Tutorial-9.Lanczos_Block_Lanczos.jl",
    "src/documentation/2.Tutorial-3.Parallelization_Basics.jl",
    "src/documentation/2.Tutorial-4.Optimized_Mapping.jl",
    "src/documentation/2.Tutorial-5.Reducing_Memory_Access.jl",
    "src/documentation/2.Tutorial-7.Hermitian_matrices.jl",
    "src/documentation/2.Tutorial-6.Working_with_GPUs.jl",

    # Advanced tutorials (XXZ)
    "src/documentation/3.Advanced-Optimization_XXZ.jl",
    "src/documentation/3.Advanced-1.XXZ_Baseline.jl",
    "src/documentation/3.Advanced-2.XXZ_Custom_Mapping.jl",
    "src/documentation/3.Advanced-3.XXZ_Rescaled_Clenshaw.jl",
    "src/documentation/3.Guidelines.jl",

    # Models
    "src/documentation/4.Quantum_Sun_QSun.jl",
    "src/documentation/4.Models-2.XXZ.jl",
    "src/documentation/4.Models-3.J1J2.jl",

    # Documentation section (docstrings)
    "src/documentation/5.Docs-1.Functions.jl",
    "src/documentation/5.Docs-4.Models.jl",
    "src/documentation/5.Docs-2.Configs_Parallelization.jl",
    "src/documentation/5.Docs-3.Reports.jl",

    "src/documentation/6.FAQ.jl",
]

generated_dir = "src/documentation/generated"
mkpath(generated_dir)

for file in files_to_literate
    Literate.markdown(file, generated_dir)
end

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
        "Home" => "documentation/generated/0.Home.md",
        "Getting Started" => "documentation/generated/1.Getting_Started.md",

        "Tutorials" => [
            "Beginner" => [
                "Choosing Target" => "documentation/generated/2.Tutorial-2.Choosing_Target.md",
                "Reporting" => "documentation/generated/2.Tutorial-8.Reporting.md",
                "Lanczos and Block Lanczos Factorization" => "documentation/generated/2.Tutorial-9.Lanczos_Block_Lanczos.md",
                "Parallelization" => "documentation/generated/2.Tutorial-3.Parallelization_Basics.md",
                "Optimized Mapping" => "documentation/generated/2.Tutorial-4.Optimized_Mapping.md",
                "Reducing Memory Access" => "documentation/generated/2.Tutorial-5.Reducing_Memory_Access.md",
                "Hermitian matrices" => "documentation/generated/2.Tutorial-7.Hermitian_matrices.md",
                "Working with GPUs" => "documentation/generated/2.Tutorial-6.Working_with_GPUs.md",
            ],
            "Advanced" => [
                "Optimization of the XXZ Model" => "documentation/generated/3.Advanced-Optimization_XXZ.md",
                "Custom Mapping" => "documentation/generated/3.Advanced-2.XXZ_Custom_Mapping.md",
                "Automatic Optimization" => "documentation/generated/3.Advanced-1.XXZ_Baseline.md",
                "GPU Implementation" => "documentation/generated/3.Advanced-3.XXZ_Rescaled_Clenshaw.md",
            ],
        ],

        "Models" => [
            "Quantum Sun (QSun)" => "documentation/generated/4.Quantum_Sun_QSun.md",
            "XXZ" => "documentation/generated/4.Models-2.XXZ.md",
            "J1-J2" => "documentation/generated/4.Models-3.J1J2.md",
        ],

        "Guidelines" => "documentation/generated/3.Guidelines.md",

        "Documentation" => [
            "Core Functions" => "documentation/generated/5.Docs-1.Functions.md",
            "Models" => "documentation/generated/5.Docs-4.Models.md",
            "Configs and Parallelization Types" => "documentation/generated/5.Docs-2.Configs_Parallelization.md",
            "Reports, Logging, and Defaults" => "documentation/generated/5.Docs-3.Reports.md",
        ],

        "FAQ / Troubleshooting" => "documentation/generated/6.FAQ.md",
    ],
)

deploydocs(
    repo="github.com/RockClimbingRocks/Polfed.jl.git",
    devbranch="main",
)
