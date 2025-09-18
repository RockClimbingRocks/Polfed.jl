
abstract type Parallelization end

mutable struct NoParallel<:Parallelization end
mutable struct MulColsParallel<:Parallelization end
mutable struct TwoLevelParallel<:Parallelization nt_per_col::Int end