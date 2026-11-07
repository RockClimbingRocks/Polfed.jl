# The consistency check every built kernel passes before it reaches the solver.
#
# Shared by all patterns rather than duplicated inside each: what it verifies — "does this closure
# compute the same thing as `mul!`?" — has nothing to do with which pattern produced the closure.
# Each pattern's own validate_kernel.jl holds only what IS pattern-specific: `can_apply`/`why_not`.
#
# Why on every build, not only in tests: a row kernel walks index tables by hand. A mistake there
# does not crash — it returns a slightly wrong matvec, and POLFED converges smoothly to the wrong
# eigenpairs. Two matvecs against the thousands a run performs is a cheap way to make that
# impossible.

"""
    verify_kernel(f!, H, a, b, ::Val{bs}, name; rtol=1e-10) -> true

Check that the built row-layout closure `f!` reproduces the rescaled operator
``\\tilde H = (H - b)/a`` on one random `bs × dim` block, measured against plain `mul!`.

Returns `true`, or throws an `ErrorException` naming the kernel and the deviation. The tolerance
is relative to the reference block's magnitude, so it is meaningful at any scale.

# Arguments
- `f!`: the closure under test, `(Y, X) -> nothing` with `Y`, `X` of size `bs × dim`.
- `H`: the original matrix.
- `a`, `b`: the rescaling the closure is supposed to implement.
- `name`: the kernel's name, used in the error message.
- `rtol`: relative tolerance.
"""
function verify_kernel(f!, H::AbstractMatrix{T}, a::Real, b::Real, ::Val{bs}, name;
                       rtol::Real = 1e-10) where {T,bs}
    D = size(H, 1)
    X = rand(T, bs, D)
    Y = similar(X)
    f!(Y, X)

    Xc = permutedims(X)                       # reference in column form, then back
    want = permutedims((H * Xc .- T(b) .* Xc) ./ T(a))

    scale = max(maximum(abs, want), one(real(T)))
    err = maximum(abs, Y .- want)
    err <= rtol * scale || error(
        "verify_kernel: kernel :$(name) disagrees with mul! on the rescaled operator — " *
        "max deviation $(err) exceeds rtol*scale = $(rtol * scale). The kernel is not computing " *
        "(H - b)/a; results from it would be silently wrong.")
    return true
end
