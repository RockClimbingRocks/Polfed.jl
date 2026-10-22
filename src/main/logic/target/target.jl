


const exeptable_energy_positions = (:left, :right, :center)

"""
    Target(rescaled_energy, howmany, rescaled_energy_position)
"""
struct Target
    rescaled_energy::Float64
    howmany::Int
    rescaled_energy_position::Symbol

    function Target(rescaled_energy::Real, howmany::Int, rescaled_energy_position::Symbol)
        rescaled_energy_position ∈ exeptable_energy_positions ||
            throw(ArgumentError("Invalid rescaled_energy_position: $rescaled_energy_position (expected one of $(exeptable_energy_positions))"))
        
            new(float(rescaled_energy), howmany, rescaled_energy_position)
    end
end


include("target_spec.jl")
include("resolve_target.jl")
include("resolve_window.jl")

