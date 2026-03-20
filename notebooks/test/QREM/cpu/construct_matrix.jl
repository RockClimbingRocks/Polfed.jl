
"""
    construct_matrix(model::QREM) -> SparseMatrixCSC

Constructs the σₓ (Sₓ for arbitrary spin) part of the QREM Hamiltonian.
"""
function construct_matrix(model::QREM; pu::String="cpu")
    L = model.L
    S = model.spin
    hx = model.hx

    @assert S > 0 "Spin must be positive"
    d = Int(2S + 1)         # Local Hilbert space dimension
    D = d^L                 # Total Hilbert space size

    # Construct the total Hamiltonian H = -hx * sum_i Sx^(i)
    id = spdiagm(0 => ones(d))
    Sx = sparse(build_Sx(S))
    ops = [id for _ in 1:L]
    ops[1] = Sx


    H = spzeros(Float64, D, D)
    for _ in 1:L
        H += hx * kron(ops...)
        circshift!(ops,1)
    end

    H += spdiagm(0 => model.diags)

    pu == "cpu" && return H
    pu == "gpu" && return CUDA.CUSPARSE.CuSparseMatrixCSR(H)
    throw(error("Unknown processing unit: $pu"))
end

"""
    build_Sx(S::Real) -> Matrix

Returns the Sₓ operator for spin S.
"""
function build_Sx(s::Real)
    d = Int(2s + 1)
    Sx = zeros(Float64, d, d)
    for m_idx in 1:d
        m = s - (m_idx - 1)
        # Lowering
        if m > -s
            coeff = 0.5 * sqrt((s + m) * (s - m + 1))
            Sx[m_idx, m_idx + 1] = coeff
        end
        # Raising
        if m < s
            coeff = 0.5 * sqrt((s - m) * (s + m + 1))
            Sx[m_idx, m_idx - 1] = coeff
        end
    end
    return Sx
end


"""
    build_Sz(S::Real) -> Matrix

Returns the Sz operator for spin S.
"""
function build_Sz(s::Real)
    d = Int(2s + 1)
    Sz = zeros(Float64, d, d)
    for m_idx in 1:d
        m = s - (m_idx - 1)
        Sz[m_idx, m_idx] = m
    end
    return Sz
end
