//! Octree y Barnes-Hut para simulaciones N-body en **gadget-ng**.
pub mod barnes_hut;
pub mod octree;
#[cfg(feature = "simd")]
pub mod rayon_bh;

pub use barnes_hut::BarnesHutGravity;
pub use octree::{
    walk_stats_begin, walk_stats_end, OctNode, Octree, WalkStats, NO_CHILD,
};
#[cfg(feature = "simd")]
pub use rayon_bh::RayonBarnesHutGravity;
