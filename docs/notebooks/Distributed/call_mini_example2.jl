# call_mini_example.jl

include("distributed_mini_example2.jl")
using .MiniPolfed

# println("--- First run: Requesting 1 workers for 8 tasks ---")
# # Starts with 1 process, adds 2 workers.
# MiniPolfed.run_example(8; nworkers = 1)

# println("\n\n--- Second run: Requesting 2 workers for 8 tasks ---")
# # Starts with 3 processes (1 main + 2 workers), adds 3 more to reach 5 workers.
# MiniPolfed.run_example(8; nworkers = 2)

println("\n\n--- Third run: Requesting 4 workers for 8 tasks ---")
# Starts with 6 processes (1 main + 5 workers), removes 2 to reach 3 workers.
MiniPolfed.run_example(8; nworkers = 4)

# println("\n\n--- Third run: Requesting 4 workers for 8 tasks ---")
# # Starts with 6 processes (1 main + 5 workers), removes 2 to reach 3 workers.
# MiniPolfed.run_example(8; nworkers = 8)

