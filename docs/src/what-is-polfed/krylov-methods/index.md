# Krylov Methods

Exact diagonalization gives direct access to the eigenvalues and eigenvectors of a Hamiltonian, but it becomes prohibitively expensive as the Hilbert-space dimension grows. For a matrix of dimension ``\mathcal{D}``, full diagonalization typically scales as ``\mathcal{O}(\mathcal{D}^3)`` in time and requires storing dense objects. This quickly becomes impossible for many-body systems, where ``\mathcal{D}`` often grows exponentially with system size.

In many quantum problems, however, the Hamiltonian is sparse. Each basis state is connected only to a relatively small number of other basis states. Krylov methods exploit this structure by avoiding full diagonalization. Instead, they use repeated applications of the Hamiltonian to vectors. This makes them one of the central tools for extracting a small number of eigenpairs from large sparse matrices.

## Krylov subspace

Starting from an initial vector ``\ket{\phi_0}``, one can generate the Krylov subspace

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

Instead of diagonalizing the full Hamiltonian ``H``, Krylov methods project the problem onto this much smaller subspace. If the Krylov subspace captures the relevant part of the spectrum well enough, the eigenvalues of the projected matrix approximate selected eigenvalues of the original Hamiltonian.

The important point is that the Hamiltonian does not need to be stored as a dense matrix. We only need a way to compute the matrix-vector product

```math
\ket{w} = H\ket{v}.
```

This is why Krylov methods are particularly useful for large sparse Hamiltonians.

## Lanczos algorithm

For Hermitian matrices, the most common Krylov method is the Lanczos algorithm. Starting from a normalized vector ``\ket{\phi_0}``, the algorithm constructs an orthonormal basis of Krylov vectors,

```math
\left\{
\ket{\phi_0},
\ket{\phi_1},
\dots,
\ket{\phi_m}
\right\}.
```

In this basis, the Hamiltonian is represented by a much smaller tridiagonal matrix, called the Lanczos matrix. The Lanczos recurrence has the form

```math
H\ket{\phi_i}
=
b_i \ket{\phi_{i-1}}
+
a_i \ket{\phi_i}
+
b_{i+1}\ket{\phi_{i+1}}.
```

Here, ``a_i`` and ``b_i`` define the diagonal and offdiagonal entries of the tridiagonal Lanczos matrix,

```math
H_L =
\begin{pmatrix}
a_0 & b_1  &        &        \\
b_1 & a_1  & b_2    &        \\
    & b_2  & a_2    & \ddots \\
    &      & \ddots & \ddots
\end{pmatrix}.
```

The eigenvalues of ``H_L`` are called Ritz values. They approximate eigenvalues of the original Hamiltonian ``H``. Since ``H_L`` is much smaller than ``H``, it can be diagonalized cheaply.

The corresponding approximate eigenvectors are expressed in the Krylov basis. If ``v_{ij}`` is an eigenvector component of ``H_L``, then the approximate eigenstate of ``H`` has the form

```math
\ket{\psi_j}
=
\sum_i v_{ij}\ket{\phi_i}.
```

In exact arithmetic, the Lanczos vectors would remain mutually orthogonal. In numerical calculations, however, finite precision can lead to a gradual loss of orthogonality. For accurate eigenvectors, especially when many eigenpairs are requested, reorthogonalization is often required.

## Why Lanczos finds spectral edges efficiently

The Lanczos algorithm is especially efficient for finding extremal eigenvalues, meaning eigenvalues near the lower or upper edge of the spectrum. Intuitively, repeated applications of ``H`` allow the Krylov subspace to build polynomials of the Hamiltonian. These polynomials can strongly amplify isolated extremal eigenvalues while suppressing the rest of the spectrum.

This is why Lanczos methods are very effective for ground-state calculations or for extracting the largest eigenvalues of a matrix. However, eigenvalues in the middle of the spectrum are usually much harder to isolate. They are surrounded by many nearby eigenvalues, so the convergence is typically much slower.

This is the central motivation for using a spectral transformation in POLFED. Instead of applying Lanczos directly to the original Hamiltonian, POLFED first applies a polynomial filter that makes the desired mid-spectrum eigenpairs appear as extremal eigenpairs of the transformed problem.

For a more detailed discussion of Krylov methods, Lanczos convergence, block Lanczos, and their role in POLFED, see the accompanying article:

- SciPost submission: <https://scipost.org/submissions/2605.10191v2/>
- arXiv preprint: <https://arxiv.org/abs/2605.10191>