# [Krylov background](@id krylov_background)

Why edges are easy, the middle is hard, and what POLFED does about it.

Exact diagonalization gives direct access to every eigenvalue and eigenvector of a
Hamiltonian, but it becomes prohibitively expensive as the Hilbert-space dimension grows.
For a matrix of dimension ``\mathcal{D}``, full diagonalization typically scales as
``\mathcal{O}(\mathcal{D}^3)`` in time and requires storing dense objects. This quickly
becomes impossible for many-body systems, where ``\mathcal{D}`` often grows exponentially
with system size.

In many quantum problems, however, the Hamiltonian is sparse: each basis state is connected
to only a small number of other basis states. Krylov methods exploit this by never
diagonalizing anything of size ``\mathcal{D}`` — they use repeated applications of the
Hamiltonian to vectors instead.

## The Krylov subspace

Starting from an initial vector ``\ket{\phi_0}``, one generates

```math
\mathcal{K}_m(H,\ket{\phi_0})
=
\mathrm{span}
\left\{
\ket{\phi_0},
H\ket{\phi_0},
H^2\ket{\phi_0},
\dots,
H^{m-1}\ket{\phi_0}
\right\}.
```

The problem is projected onto this much smaller subspace. If the subspace captures the
relevant part of the spectrum, the eigenvalues of the projected matrix approximate selected
eigenvalues of ``H``.

The important consequence: the Hamiltonian never needs to exist as a dense matrix. All that
is required is a way to compute

```math
\ket{w} = H\ket{v}.
```

This is exactly why POLFED accepts a bare operator `f!(Y, X)` as readily as a matrix — see
[Bring your own operator](@ref operator_interface).

## The Lanczos algorithm

For Hermitian matrices the standard Krylov method is Lanczos. From a normalized
``\ket{\phi_0}`` it builds an orthonormal basis
``\{\ket{\phi_0}, \ket{\phi_1}, \dots, \ket{\phi_m}\}`` in which ``H`` is represented by a
small tridiagonal matrix. The recurrence is

```math
H\ket{\phi_i}
=
b_i \ket{\phi_{i-1}}
+
a_i \ket{\phi_i}
+
b_{i+1}\ket{\phi_{i+1}},
```

with ``a_i`` and ``b_i`` the diagonal and offdiagonal entries of

```math
H_L =
\begin{pmatrix}
a_0 & b_1  &        &        \\
b_1 & a_1  & b_2    &        \\
    & b_2  & a_2    & \ddots \\
    &      & \ddots & \ddots
\end{pmatrix}.
```

The eigenvalues of ``H_L`` are the **Ritz values**; they approximate eigenvalues of ``H``
and are cheap to compute because ``H_L`` is small. The approximate eigenvectors live in the
Krylov basis: if ``v_{ij}`` is an eigenvector component of ``H_L``, then

```math
\ket{\psi_j}
=
\sum_i v_{ij}\ket{\phi_i}.
```

In exact arithmetic the Lanczos vectors stay mutually orthogonal. In floating point they
gradually lose orthogonality, so reorthogonalization is needed for accurate eigenvectors —
especially when many eigenpairs are requested. POLFED reorthogonalizes fully by default;
see [`FullRO`](@ref UniPolfed.Numerics.Lanczos.FullRO) and its alternatives.

## Why Lanczos finds spectral edges efficiently

Repeated applications of ``H`` let the Krylov subspace build *polynomials* of the
Hamiltonian, and those polynomials strongly amplify isolated extremal eigenvalues while
suppressing the rest. That is why Lanczos is so effective for ground states and for the
largest eigenvalues.

Eigenvalues in the middle of the spectrum are a different story. They sit in a dense crowd
of nearby eigenvalues with no gap to exploit, and convergence becomes slow — often
hopelessly so for many-body spectra.

**This is the entire motivation for POLFED.** Rather than running Lanczos on ``H`` and
hoping, POLFED first applies a polynomial filter that makes the desired mid-spectrum
eigenpairs *extremal* in the transformed problem — putting them back in the regime where
Lanczos is fast. See [Filters](@ref filters).

## Block Lanczos

Passing a matrix `x0` instead of a vector selects **block Lanczos**, with block size equal
to the number of columns. Instead of one Krylov vector per iteration, the algorithm carries
a block of them.

This buys two different things.

**Degenerate and clustered eigenvalues.** A single starting vector spans a one-dimensional
space, so plain Lanczos can return only *one* eigenvector per eigenvalue — it cannot resolve a
degenerate pair at all, and it converges slowly when eigenvalues are tightly clustered. A block
of size ``b`` can resolve multiplicities up to ``b``. That matters here: symmetry-reduced
many-body spectra routinely carry exact degeneracies, and POLFED targets the dense middle of the
spectrum, where near-degenerate clusters are the norm rather than the exception.

**Hardware utilization.** The filter is applied to several vectors at once, turning a
memory-bound matrix-vector product into a matrix-matrix product with far better arithmetic
intensity, and exposing independent columns that threads and GPUs can work on in parallel.

It is not free. The projected problem grows with the block size. The estimate for the required
Krylov dimension is an empirical fit for the **iteration count**, taken at the
eigenpairs-per-block ratio `x = howmany / blocksize` and multiplied back by the block size,

```julia
expected_krylov_dim(filter, howmany, blocksize, η) =
    ceil(Int64, 16.725632 * x^0.40598213 * (1 + x / 50.265139)^(1 - 0.40598213) * blocksize * η)
```

and since that fit is sublinear in `x`, a larger block buys fewer iterations than it adds
vectors: the dimension grows with `blocksize` (at `howmany = 1000`, `η = 1`: `1681`, `2080`,
`2318` for `blocksize = 1, 10, 16`). An oversized block enlarges the projected problem and slows
convergence rather than helping. The practical rule of thumb is

```math
\frac{\texttt{howmany}}{\texttt{block\_size}} \gtrsim 100 .
```

It is the ratio that matters, not the block size on its own. Tuning advice lives in
[Parallelization](@ref parallelization) and [Guidelines](@ref guidelines).

## Going further

For a fuller discussion of Krylov methods, Lanczos convergence, block Lanczos, and their
role in POLFED, see the accompanying article:

- SciPost submission: <https://scipost.org/submissions/2605.10191v2/>
- arXiv preprint: <https://arxiv.org/abs/2605.10191>
