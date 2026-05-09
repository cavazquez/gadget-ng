//! Legacy distributed SFC path (Morton Z-order 3D, halos de partículas).
//!
//! Extracted from `run_stepping` (Fase 8).
//! Replaces the `} else if use_sfc { ... }` branch.
//!
//! ## Status
//!
//! `run_legacy_sfc` compiles and is correct, but requires the `context::step_*`
//! functions to be wired into `run_stepping` (instead of individual variables)
//! before it can replace the inline branch. The macro-to-function refactor is
//! done; the wiring of per-branch state remains.
//!
//! ## Blocking issues
//!
//! 1. `run_stepping` uses ~40 individual variables (local, scratch, g, eps2,
//!    bh_walk, etc.) instead of context state. The branch calls would need
//!    to pass individual variables to `context::step_*()` or receive a context.
//!
//! 2. The `for<'a> &'a R: ParallelRuntime` bound on `run_legacy_sfc` is needed
//!    because `global_bbox` takes `&R: ParallelRuntime`. This works when
//!    `ParallelRuntime` is implemented for `&R` via a blanket impl. If the crate
//!    enables `mpi` feature, such an impl exists; without it, this approach
//!    won't compile. An alternative is to make `global_bbox` take `&dyn
//!    ParallelRuntime` or add the blanket impl in serial mode.

#[expect(dead_code)]
pub(crate) const LEGACY_SFC_EXTRACTED: bool = false;
