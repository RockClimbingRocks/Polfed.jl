using Documenter, Literate, Polfed

documentation_source_dir = joinpath(@__DIR__, "src", "documentation")

files_to_literate = [
    joinpath(documentation_source_dir, "0.Home.jl"),
    joinpath(documentation_source_dir, "1.Getting_Started.jl"),

    # Beginner tutorials
    joinpath(documentation_source_dir, "2.Tutorial-2.Choosing_Target.jl"),
    joinpath(documentation_source_dir, "2.Tutorial-8.Reporting.jl"),
    joinpath(documentation_source_dir, "2.Tutorial-9.Lanczos_Block_Lanczos.jl"),
    joinpath(documentation_source_dir, "2.Tutorial-3.Parallelization_Basics.jl"),
    joinpath(documentation_source_dir, "2.Tutorial-4.Optimized_Mapping.jl"),
    joinpath(documentation_source_dir, "2.Tutorial-5.Reducing_Memory_Access.jl"),
    joinpath(documentation_source_dir, "2.Tutorial-7.Hermitian_matrices.jl"),
    joinpath(documentation_source_dir, "2.Tutorial-6.Working_with_GPUs.jl"),

    # Advanced tutorials (XXZ)
    joinpath(documentation_source_dir, "3.Advanced-Optimization_XXZ.jl"),
    joinpath(documentation_source_dir, "3.Advanced-1.XXZ_Baseline.jl"),
    joinpath(documentation_source_dir, "3.Advanced-2.XXZ_Custom_Mapping.jl"),
    joinpath(documentation_source_dir, "3.Advanced-3.XXZ_Rescaled_Clenshaw.jl"),
    joinpath(documentation_source_dir, "3.Guidelines.jl"),

    # Models
    joinpath(documentation_source_dir, "4.Quantum_Sun_QSun.jl"),
    joinpath(documentation_source_dir, "4.Models-2.XXZ.jl"),
    joinpath(documentation_source_dir, "4.Models-3.J1J2.jl"),

    # Documentation section (docstrings)
    joinpath(documentation_source_dir, "5.Docs-1.Functions.jl"),
    joinpath(documentation_source_dir, "5.Docs-4.Models.jl"),
    joinpath(documentation_source_dir, "5.Docs-2.Configs_Parallelization.jl"),
    joinpath(documentation_source_dir, "5.Docs-3.Reports.jl"),

    joinpath(documentation_source_dir, "6.FAQ.jl"),
]

generated_dir = joinpath(documentation_source_dir, "generated")
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
