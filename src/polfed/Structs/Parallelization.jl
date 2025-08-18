
abstract type Parallelization end

mutable struct NoParallel<:Parallelization end
mutable struct MulColsParallel<:Parallelization end
