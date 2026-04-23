---
title: 'gadget-ng: A modular N-body/SPH cosmological simulation code in Rust'
tags:
  - Rust
  - cosmology
  - N-body
  - SPH
  - radiative transfer
  - MPI
authors:
  - name: Cristian Alvarez-Vazquez
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent
    index: 1
date: 2026-04-23
bibliography: paper.bib
---

# Summary

`gadget-ng` is a modular cosmological simulation framework written in Rust,
designed for N-body and hydrodynamical (SPH) simulations of large-scale structure
formation. It implements a TreePM gravity solver, Smoothed Particle Hydrodynamics
(SPH) with cosmological integration, a hierarchical AMR Particle-Mesh solver, M1
moment-based radiative transfer, non-equilibrium hydrogen and helium chemistry,
and distributed-memory parallelism via MPI. The code is organized as a Rust
workspace of specialized crates, enabling modular development, strong compile-time
safety guarantees, and zero-cost abstractions for high-performance computing.

# Statement of Need

Modern cosmological simulations require simultaneously handling gravitational
dynamics (N-body), gas physics (hydrodynamics, cooling, feedback), radiative
transfer (UV background, reionization), and non-equilibrium chemistry. Existing
production codes such as GADGET-4 [@springel2021simulating] and RAMSES
[@teyssier2002cosmological] are written in C/C++ and Fortran, where memory
safety, concurrency bugs, and undefined behavior are persistent challenges.

`gadget-ng` addresses these limitations by leveraging Rust's ownership system,
which enforces memory safety without a garbage collector and eliminates entire
classes of bugs (data races, use-after-free, buffer overflows) at compile time.
The code is designed as a research platform for algorithm development, where
the modular crate structure makes it straightforward to add new physics modules
without modifying the core simulation loop.

Key distinguishing features:

- **Memory safety by construction**: Rust's borrow checker eliminates undefined behavior
  common in C-based codes (out-of-bounds access, race conditions).
- **Modular workspace**: 16 specialized crates from `gadget-ng-core` (data structures)
  to `gadget-ng-rt` (radiative transfer) that can be used independently.
- **Full physics stack**: TreePM + SPH + AMR-PM + RT M1 + non-equilibrium chemistry
  in a single unified framework.
- **Dual GPU support**: experimental CUDA and HIP (ROCm) backends for GPU-accelerated
  gravity via a second build chain.

# Algorithms

## Gravitational Dynamics

`gadget-ng` implements three gravity solvers:

**Direct N-body** (`gadget-ng-gpu`): O(N²) force calculation using a WGSL compute
shader via the `wgpu` library. Supports GPU acceleration with automatic CPU fallback.

**Tree-PM** (`gadget-ng-tree`, `gadget-ng-pm`): Short-range gravitational forces
are computed with a Barnes-Hut octree [@barnes1986hierarchical] using a
monopole-quadrupole opening criterion. Long-range forces are computed on a regular
PM grid using 3D FFT (rustfft). The tree and PM components are split at a scale
$r_s$ determined by the PM grid resolution. In distributed (MPI) mode, a
Locally Essential Tree (LET) is exchanged between ranks to enable correct
short-range force calculations without full particle replication.

**Hierarchical AMR-PM** (`gadget-ng-pm`): An adaptive multi-level PM solver
identifies high-density regions and places refinement patches. The base PM grid
and each patch are solved independently with Poisson's equation via FFT, and
forces are superposed with appropriate interpolation. N-level recursion is
supported via `build_amr_hierarchy`.

## Time Integration

Multiple integrators are available:

- **Leapfrog KDK**: second-order symplectic, cosmological and non-cosmological variants.
- **Yoshida 4th order**: 4th-order symplectic using the Yoshida coefficients.
- **Hierarchical block timesteps** (`gadget-ng-integrators`): particles are
  grouped into timestep bins (powers of 2); only the bin appropriate for each
  particle's acceleration is integrated at each macro-step. This dramatically
  reduces force evaluations for clustered configurations.

## SPH Hydrodynamics

`gadget-ng-sph` implements the density-entropy SPH formulation [@springel2002cosmological]
with:

- Adaptive smoothing length via Newton iteration (50±5 neighbor target).
- Wendland $C^4$ kernel.
- Cosmological KDK integration with scale factor $a(t)$.
- Stochastic supernova feedback: star formation rate from the Schmidt-Kennicutt
  law, stochastic SN kicks with $v_\mathrm{kick} = 350\,\mathrm{km/s}$.
- Atomic H+He radiative cooling.

## Radiative Transfer

`gadget-ng-rt` implements the M1 closure model [@levermore1984relating;@rosdahl2013ramses]:

$$
\partial_t E + \nabla \cdot \mathbf{F} = \eta - c\kappa_\mathrm{abs} E
$$
$$
\partial_t \mathbf{F} + c^2 \nabla \cdot \mathbf{P} = -c\kappa_\mathrm{abs}\mathbf{F}
$$

where the Eddington tensor is closed as $\mathbf{P} = f(\xi) E \hat{\mathbf{n}}\hat{\mathbf{n}}$
with the M1 closure factor $f(\xi) = (3 + 4\xi^2)/(5 + 2\sqrt{4-3\xi^2})$.

The solver uses an HLL Riemann scheme [@harten1983upstream] on a Cartesian grid with
reduced speed of light $c_\mathrm{red} = c / f_\mathrm{red}$ (typically $f_\mathrm{red} = 100$).

## Non-Equilibrium Chemistry

The chemistry module (`gadget-ng-rt::chemistry`) solves the coupled rate equations
for 6 species (HI, HII, HeI, HeII, HeIII, e$^-$) using an implicit subcycled solver
[@anninos1997cosmological]:

- Recombination rates: Verner & Ferland (1996) fits [@verner1996atomic].
- Collisional ionization rates: Cen (1992) [@cen1992hydrodynamic].
- Photoionization rates coupled to the M1 radiation field.
- Approximate Ly$\alpha$ + bremsstrahlung cooling.

## MPI Parallelism

`gadget-ng-parallel` implements the parallel runtime via `rsmpi` (Rust MPI bindings).
The domain decomposition uses a space-filling curve (Hilbert or Morton), and load
balancing is adaptive based on measured tree-walk cost per particle.

For the PM solver, a slab decomposition is used in Y (`SlabDecomposition`), with
halo exchange between neighboring ranks for the FFT pencils. For the radiative
transfer field, a similar Y-slab decomposition (`RadiationFieldSlab`) is used with
ghost cell exchange (`exchange_radiation_halos_mpi`). For the AMR patches, rank 0
coordinates global patch identification after an allreduce of the density grid
(`broadcast_patch_forces_mpi`).

# Validation

## Power Spectrum P(k)

The linear matter power spectrum is validated against the Eisenstein-Hu fitting
formula [@eisenstein1998baryonic]. Simulations starting from 2LPT initial conditions
at $z=49$ and integrated to $z=0$ reproduce $\sigma_8$ to within 2% for
$N=128^3$ particle runs in a $L=100\,h^{-1}\mathrm{Mpc}$ box (Phase 79 validation suite).

## Halo Mass Function

The friends-of-friends (FoF) halo mass function is compared against the
Tinker et al. (2008) analytic fit [@tinker2008toward], showing agreement within
10% over two decades in mass for $N=128^3$ runs.

## Strömgren Sphere

The M1 radiative transfer is validated against the analytic Strömgren sphere
solution: for a point UV source with ionization rate $\dot{N}_\mathrm{ion}$
in a uniform medium of density $n_H$, the equilibrium radius is:
$$
R_S = \left(\frac{3\dot{N}_\mathrm{ion}}{4\pi n_H^2 \alpha_B}\right)^{1/3}
$$
The code reproduces $R_S$ to better than 5% for typical simulation parameters.

## NFW Profiles

Halo density profiles are fit to the NFW form [@navarro1997universal]:
$$
\rho(r) = \frac{\rho_s}{(r/r_s)(1 + r/r_s)^2}
$$
The concentration-mass relation $c(M)$ is compared against the Ludlow et al. (2016)
model [@ludlow2016mass] (Phase 58).

# Performance

## TreePM Scaling

Weak scaling tests show near-linear scaling up to 8 MPI ranks for $N_\mathrm{loc} = 32^3$
particles per rank, with communication overhead below 20% for the LET exchange.

## GPU Acceleration

The direct gravity kernel (`GpuDirectGravity`) achieves a 5–15× speedup over
single-thread CPU for $N = 1000$ particles on a desktop GPU, with the GPU
advantage growing as $O(N^2)$ costs dominate at larger N.

## Benchmarks

Benchmarks are available via:
```bash
cargo bench -p gadget-ng-gpu --bench gpu_vs_cpu
```
Results are stored in `target/criterion/` as HTML reports.

# Acknowledgments

The code structure was inspired by GADGET-4 [@springel2021simulating] and RAMSES
[@teyssier2002cosmological]. The M1 radiative transfer implementation follows
RAMSES-RT [@rosdahl2013ramses]. The non-equilibrium chemistry solver follows
[@anninos1997cosmological].

# References
