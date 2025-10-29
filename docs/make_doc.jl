
using Documenter, Literate, Polfed


files_to_leterate = [
    "src/documentation/1.Tutorial-1.My_first_POLFED_run.jl", 
    "src/documentation/1.Tutorial-2.Knowing_Your_Parallelization.jl",
    "src/documentation/1.Tutorial-3.Constructing_Optimized_Mapping.jl",
    "src/documentation/1.Tutorial-4.Reducing_Memory_Access.jl",

    "src/documentation/2.Docs-Reporting.jl",
    "src/documentation/2.Docs-Parallelization.jl",
    "src/documentation/2.Docs-Polfed.jl",
    "src/documentation/2.Docs-PolfedDefaults.jl",
]
for file in files_to_leterate
    # Convert Literate scripts into Markdown
    Literate.markdown(file, "src/documentation/generated")
end



makedocs(
    modules=[Polfed],
    authors="RockClimbingRocks <rok123.pintar2@gmail.com> and contributors",
    sitename="Polfed.jl",
    checkdocs = :none,
    format=Documenter.HTML(;
        canonical="https://RockClimbingRocks.github.io/Polfed.jl",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",

        "Tutorial" => [
            "My first POLFED run" => "documentation/generated/1.Tutorial-1.My_first_POLFED_run.md",
            "Knowing your parallelization" => "documentation/generated/1.Tutorial-2.Knowing_Your_Parallelization.md",
            "Constructing optimized mapping" => "documentation/generated/1.Tutorial-3.Constructing_Optimized_Mapping.md",
            "Reducing memory access" => "documentation/generated/1.Tutorial-4.Reducing_Memory_Access.md",
        ],

        "Documentation" => [
            "Polfed" => "documentation/generated/2.Docs-Polfed.md",
            "Polfed defaults" => "documentation/generated/2.Docs-PolfedDefaults.md",
            "Reporting" => "documentation/generated/2.Docs-Reporting.md",
            "Parallelization" => "documentation/generated/2.Docs-Parallelization.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/RockClimbingRocks/Polfed.jl.git",
    devbranch="main",
)