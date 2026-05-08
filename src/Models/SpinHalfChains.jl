const _SUPPORTED_BOUNDARIES = (:open, :periodic)

"""
    xxz_hamiltonian(
        L::Integer,
        Lup::Integer,
        J::Real,
        Δ::Real,
        W::Real;
        boundary::Symbol=:periodic,
        field::Real=0.0,
        fields=nothing,
        rng::AbstractRNG=Random.default_rng(),
        use_sparse::Bool=true,
    )

Construct a spin-1/2 XXZ-chain Hamiltonian,

```math
H = J \\sum_{i=1}^{L}
    \\left(
        S_i^x S_{i+1}^x
        + S_i^y S_{i+1}^y
        + \\Delta S_i^z S_{i+1}^z
    \\right)
    + \\sum_i h_i S_i^z.
```

where the constructor works in a fixed spin-1/2 magnetization sector.

# Positional Arguments

- `L::Integer`:
  Chain length.
- `Lup::Integer`:
  Number of spin-up sites. This fixes the magnetization sector through
  ``S_z = Lup - L/2``.
- `J::Real`:
  Nearest-neighbor exchange scale.
- `Δ::Real`:
  XXZ anisotropy parameter.
- `W::Real`:
  Disorder width for longitudinal fields.

# Keyword Arguments

- `boundary::Symbol=:periodic`:
  Boundary condition. Supported values are `:periodic` and `:open`.
- `field::Real=0.0`:
  Center of the disorder window. When random fields are generated internally,
  they are sampled from `[field - W, field + W]`.
- `fields=nothing`:
  Optional explicit field realization. If given, it should be an
  `AbstractVector{<:Real}` of length `L`, and it is used directly instead of
  sampling from `W`.
- `rng::AbstractRNG=Random.default_rng()`:
  Random-number generator used when the field realization is sampled
  internally.
- `use_sparse::Bool=true`:
  If `true`, return a sparse matrix. If `false`, return a dense matrix.

# Returns

- `SparseMatrixCSC{Float64,Int}` when `use_sparse=true`.
- `Matrix{Float64}` when `use_sparse=false`.

# Notes

- This constructor currently targets spin-1/2 chains only.
- With `boundary=:periodic`, site indices wrap around exactly as written in the
  defining sum.
- With `boundary=:open`, boundary-crossing terms are omitted.
"""
function xxz_hamiltonian(
    L::Integer,
    Lup::Integer,
    J::Real,
    Δ::Real,
    W::Real;
    boundary::Symbol=:periodic,
    field::Real=0.0,
    fields=nothing,
    rng::AbstractRNG=Random.default_rng(),
    use_sparse::Bool=true,
)
    _validate_chain_length(L)
    bonds = _chain_bonds(Int(L), 1, boundary)
    local_fields = _resolve_fields(Int(L), field, W, fields, rng)

    return _spin_half_hamiltonian(
        Int(L),
        bonds,
        fill(Float64(J), length(bonds)),
        fill(Float64(Δ), length(bonds)),
        local_fields,
        Int(Lup);
        use_sparse=use_sparse,
    )
end

"""
    j1j2_hamiltonian(
        L::Integer,
        Lup::Integer,
        J1::Real,
        J2::Real,
        Δ1::Real,
        Δ2::Real,
        W::Real;
        boundary::Symbol=:periodic,
        field::Real=0.0,
        fields=nothing,
        rng::AbstractRNG=Random.default_rng(),
        use_sparse::Bool=true,
    )

Construct a spin-1/2 J1-J2 XXZ-chain Hamiltonian with nearest-neighbor and
next-nearest-neighbor couplings,

```math
H =
J_1 \\sum_{i=1}^{L}
\\left(
    S_i^x S_{i+1}^x
    + S_i^y S_{i+1}^y
    + \\Delta_1 S_i^z S_{i+1}^z
\\right)
+
J_2 \\sum_{i=1}^{L}
\\left(
    S_i^x S_{i+2}^x
    + S_i^y S_{i+2}^y
    + \\Delta_2 S_i^z S_{i+2}^z
\\right)
    + \\sum_i h_i S_i^z.
```

where both nearest-neighbor and next-nearest-neighbor terms are included in a
fixed spin-1/2 magnetization sector.

# Positional Arguments

- `L::Integer`:
  Chain length.
- `Lup::Integer`:
  Number of spin-up sites. This fixes the magnetization sector through
  ``S_z = Lup - L/2``.
- `J1::Real`:
  Nearest-neighbor exchange scale.
- `J2::Real`:
  Next-nearest-neighbor exchange scale.
- `Δ1::Real`:
  Nearest-neighbor Ising anisotropy.
- `Δ2::Real`:
  Next-nearest-neighbor Ising anisotropy.
- `W::Real`:
  Disorder width for longitudinal fields.

# Keyword Arguments

- `boundary::Symbol=:periodic`:
  Boundary condition. Supported values are `:periodic` and `:open`.
- `field::Real=0.0`:
  Center of the disorder window. When random fields are generated internally,
  they are sampled from `[field - W, field + W]`.
- `fields=nothing`:
  Optional explicit field realization. If given, it should be an
  `AbstractVector{<:Real}` of length `L`, and it is used directly instead of
  sampling from `W`.
- `rng::AbstractRNG=Random.default_rng()`:
  Random-number generator used when the field realization is sampled
  internally.
- `use_sparse::Bool=true`:
  If `true`, return a sparse matrix. If `false`, return a dense matrix.

# Returns

- `SparseMatrixCSC{Float64,Int}` when `use_sparse=true`.
- `Matrix{Float64}` when `use_sparse=false`.

# Notes

- This constructor currently targets spin-1/2 chains only.
- With `boundary=:periodic`, site indices wrap around exactly as in the sums
  above.
- With `boundary=:open`, boundary-crossing terms are omitted.
- The disorder conventions match [`xxz_hamiltonian`](@ref).
"""
function j1j2_hamiltonian(
    L::Integer,
    Lup::Integer,
    J1::Real,
    J2::Real,
    Δ1::Real,
    Δ2::Real,
    W::Real;
    boundary::Symbol=:periodic,
    field::Real=0.0,
    fields=nothing,
    rng::AbstractRNG=Random.default_rng(),
    use_sparse::Bool=true,
)
    _validate_chain_length(L)
    nearest = _chain_bonds(Int(L), 1, boundary)
    next_nearest = _chain_bonds(Int(L), 2, boundary)
    bonds = vcat(nearest, next_nearest)
    couplings = vcat(fill(Float64(J1), length(nearest)), fill(Float64(J2), length(next_nearest)))
    anisotropies = vcat(fill(Float64(Δ1), length(nearest)), fill(Float64(Δ2), length(next_nearest)))
    local_fields = _resolve_fields(Int(L), field, W, fields, rng)

    return _spin_half_hamiltonian(
        Int(L),
        bonds,
        couplings,
        anisotropies,
        local_fields,
        Int(Lup);
        use_sparse=use_sparse,
    )
end

function _validate_chain_length(L::Integer)
    L >= 2 || throw(ArgumentError("L must be at least 2; got L=$L."))
    return nothing
end

function _chain_bonds(L::Int, distance::Int, boundary::Symbol)
    boundary in _SUPPORTED_BOUNDARIES ||
        throw(ArgumentError("boundary must be one of $(_SUPPORTED_BOUNDARIES); got $boundary."))

    if boundary === :open
        distance < L || return Tuple{Int,Int}[]
        return [(i, i + distance) for i in 1:(L - distance)]
    end

    bonds = Tuple{Int,Int}[]
    for i in 1:L
        j = mod1(i + distance, L)
        i == j && continue
        push!(bonds, (i, j))
    end
    return bonds
end

function _resolve_fields(L::Int, field::Real, W::Real, fields, rng::AbstractRNG)
    if fields !== nothing
        length(fields) == L ||
            throw(ArgumentError("fields must have length L=$L; got length $(length(fields))."))
        return Float64.(collect(fields))
    end

    W = Float64(W)
    W >= 0 || throw(ArgumentError("W must be non-negative; got $W."))
    center = Float64(field)
    W == 0 && return fill(center, L)
    return center .+ W .* (2 .* rand(rng, L) .- 1)
end

function _lup_basis(L::Int, Lup::Int)
    0 <= Lup <= L || throw(ArgumentError("Lup must satisfy 0 <= Lup <= L; got Lup=$Lup for L=$L."))
    return [state for state in 0:(1 << L) - 1 if count_ones(state) == Lup]
end

@inline _spin_z(state::Int, site::Int) = ((state >> (site - 1)) & 1) == 1 ? 0.5 : -0.5
@inline _flip_pair(state::Int, i::Int, j::Int) = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))

function _spin_half_hamiltonian(
    L::Int,
    bonds::Vector{Tuple{Int,Int}},
    couplings::Vector{Float64},
    anisotropies::Vector{Float64},
    fields::Vector{Float64},
    Lup::Int;
    use_sparse::Bool,
)
    basis = _lup_basis(L, Lup)
    dim = length(basis)
    index = Dict(state => n for (n, state) in enumerate(basis))

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (col, state) in enumerate(basis)
        diag = 0.0

        for site in 1:L
            diag += fields[site] * _spin_z(state, site)
        end

        for (bond_index, (i, j)) in enumerate(bonds)
            J = couplings[bond_index]
            Δ = anisotropies[bond_index]
            zi = _spin_z(state, i)
            zj = _spin_z(state, j)

            diag += J * Δ * zi * zj

            if zi != zj
                flipped = _flip_pair(state, i, j)
                row = index[flipped]
                push!(rows, row)
                push!(cols, col)
                push!(vals, J / 2)
            end
        end

        push!(rows, col)
        push!(cols, col)
        push!(vals, diag)
    end

    H = sparse(rows, cols, vals, dim, dim)
    return use_sparse ? H : Matrix(H)
end
