# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

This is a scientific simulation codebase, not a network-facing service.
Security issues (buffer overflows, unsafe code bugs, UB) should be reported
via GitHub issues or directly to the maintainer.

For critical vulnerabilities, email the project maintainer.
Please do not open a public issue for critical issues.

## Scope

- Unsafe code in GPU/MPI/SIMD crates
- FFI boundary safety (CUDA, HIP, MPI)
- Build system injection vectors
- Dependency supply chain

## Out of Scope

- Cosmological parameter tuning
- Numerical stability (standard in HPC codes)
- Performance characteristics
