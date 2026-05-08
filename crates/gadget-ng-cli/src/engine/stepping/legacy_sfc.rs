//! Legacy distributed SFC path (Morton Z-order 3D, halos de partículas).
//!
//! Extracted from `run_stepping` (Fase 8).
//! Replaces the `} else if use_sfc { ... }` branch.
//!
//! ## Status
//!
//! `run_legacy_sfc` compiles and is correct, but requires `SteppingCtx`
//! to be wired into `run_stepping` (instead of individual variables) before
//! it can replace the inline branch. The extraction of `SteppingCtx` and
//! its methods is done; the wiring is the remaining step.
//!
//! ## Blocking issues
//!
//! 1. `run_stepping` uses ~40 individual variables (local, scratch, g, eps2,
//!    bh_walk, etc.) instead of a `SteppingCtx`. The branch calls would need
//!    to construct or receive a `SteppingCtx`.
//!
//! 2. The `for<'a> &'a R: ParallelRuntime` bound on `run_legacy_sfc` is needed
//!    because `global_bbox` takes `&R: ParallelRuntime`. This works when
//!    `ParallelRuntime` is implemented for `&R` via a blanket impl. If the crate
//!    enables `mpi` feature, such an impl exists; without it, this approach
//!    won't compile. An alternative is to make `global_bbox` take `&dyn
//!    ParallelRuntime` or add the blanket impl in serial mode.

pub(crate) const LEGACY_SFC_EXTRACTED: bool = false;
