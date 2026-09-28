# Poisson Equation FDM Solver: Serial, OpenMP, and MPI

A 2D Poisson equation solver built from scratch in C++, parallelized along
three routes, and now being restructured with modern C++ and extended toward
new simulation scenarios such as the 2D convection-diffusion equation.

## Problem

Solves the Poisson equation on the unit square with Dirichlet boundary
conditions:

```
-delta u(x, y) = f(x, y),   f(x, y) = 2*pi^2 * sin(pi*x) * cos(pi*y)
```

on a uniform N x N grid with the 5-point finite difference stencil. Interior
nodes give a symmetric positive definite linear system, stored in CRS
(compressed row storage) format, solved with the Conjugate Gradient method
until the residual norm drops by a prescribed reduction factor.

The forcing is chosen so the exact solution is known:

```
u(x, y) = sin(pi*x) * cos(pi*y)
```

Every implementation reports the discrete L2 error against this solution, so
correctness is verified directly, not just residual convergence. A grid
refinement study confirms the expected second-order accuracy of the
discretization.

## Repository Structure

The solver is being developed in two layers:

- `legacy/` contains the four original standalone programs, kept as working
  reference implementations. They share no code and are compiled directly.
- The ongoing restructuring rebuilds the solver kernels with modern C++
  (templates, concepts, and modular structure), making the discretization,
  the solver, and the parallel backends independent and reusable components.
  The convection-diffusion extension builds on this layer.

### Legacy Implementations

| File | Parallelization | How it works |
|---|---|---|
| `serial.cpp` | None | Reference implementation: CRS assembly, sparse matvec, CG |
| `openmp.cpp` | Shared memory | Same structure, with OpenMP pragmas and reductions on assembly, matvec, dot products, and vector updates |
| `mpi_matrix.cpp` | Distributed, matrix rows | Root assembles the full system and scatters row blocks with MPI_Scatterv. Each CG iteration gathers the full search direction with Allgatherv, so every process can compute its local part of the matvec |
| `mpi_domain.cpp` | Distributed, domain decomposition | No assembled matrix. A 1D Cartesian process grid decomposes the domain into row bands, each process stores its band with ghost rows, and the 5-point stencil is applied directly after a nonblocking halo exchange (Isend/Irecv + Waitall). Only scalars are reduced across processes |

The two MPI versions are deliberately different designs for the same
algorithm. The matrix-row version is simple to write but every process holds
the full iteration vectors and communicates the whole search direction each
iteration. The domain-decomposition version communicates only halo rows and
reduces only scalars, which is the structure that scales.

## Build and Run (legacy)

```
g++ -O2 -std=c++17 -Wall -o serial legacy/serial.cpp
./serial

g++ -O2 -std=c++17 -Wall -fopenmp -o openmp legacy/openmp.cpp
./openmp

mpicxx -O2 -std=c++17 -Wall -o mpi_matrix legacy/mpi_matrix.cpp
mpirun -np 4 ./mpi_matrix

mpicxx -O2 -std=c++17 -Wall -o mpi_domain legacy/mpi_domain.cpp
mpirun -np 4 ./mpi_domain
```

Each program prints iterations, initial and final residual, discrete L2
error, and solver runtime. Problem size, tolerance, and iteration limits are
constants at the top of each `main()`.

## Status

- All four legacy implementations verified against the analytical solution
- Grid-refinement study for second-order convergence
- Modern C++ restructuring of the solver kernels in progress
- Convection-diffusion equation in 2D, built on the restructured solver
- CUDA extension of the numerical kernels in progress

## References

- Shewchuk, J. R. (1994). An Introduction to the Conjugate Gradient Method
  Without the Agonizing Pain.
- Golub, G. H., Van Loan, C. F. (2013). Matrix Computations. Johns Hopkins.