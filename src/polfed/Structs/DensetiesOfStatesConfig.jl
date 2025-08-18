

mutable struct DoSConfig
    N::Int
    R::Int

    function DoSConfig(;
        N::Int=PolfedDefaults.N, 
        R::Int=PolfedDefaults.R
    )
        new(N, R)   
    end
end


mutable struct DoSConfigFull
    ρ::Union{Function, Nothing}
    kernel::Symbol
    N::Integer
    R::Integer

    function DoSConfigFull(
        dos::DoSConfig
    )
        new(nothing, :Jackson, dos.N, dos.R)   
    end
end