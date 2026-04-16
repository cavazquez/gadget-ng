//! `gadget-ng-physics` — Pruebas de validación física.
//!
//! Cada archivo en `tests/` prueba un problema clásico con solución conocida:
//!
//! | Test | Problema | Solver |
//! |------|----------|--------|
//! | `kepler_orbit` | Órbita circular de Kepler (2 cuerpos) | leapfrog KDK |
//! | `sod_shock_tube` | Tubo de Sod 1D | SPH |
//! | `plummer_virial` | Equilibrio virial en esfera de Plummer | Barnes-Hut |
