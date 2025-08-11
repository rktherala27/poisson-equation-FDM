# poisson-equation-FDM

This repository contains a collection of C++ programs that solve the two-dimensional Poisson equation on a square domain using the finite difference method. The linear system of equations resulting from the discretization is solved using the Conjugate Gradient (CG) method. The project explores different approaches to parallelization to compare their performance.

There are in total 4 files to run the same simulation with two versions using MPI, one version using OpenMP and the other serial code.

The rhs function is:

\[
f(x, y) = 2 \pi^2 \sin(\pi x) \cos(\pi y)
\],

which is used to solve the poisson equation,
\[
-\Delta u(x, y) = f(x, y)
\]

where `u(x, y)` is the unknown solution.
