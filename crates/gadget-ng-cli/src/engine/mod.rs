//! Simulation engine: integration loop, gravity kernels, checkpoints, and CLI helpers.

mod checkpoint;
mod cmds;
mod diagnostics;
mod gravity;
mod provenance;
mod stepping;
mod timings;

pub use cmds::{cmd_config_print, run_analyse, run_snapshot, run_visualize};
pub use stepping::run_stepping;
