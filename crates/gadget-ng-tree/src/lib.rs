//! Octree y Barnes-Hut para simulaciones N-body en **gadget-ng**.
pub mod barnes_hut;
mod hex_dt_patterns;
mod hexadecapole;
pub mod let_tree;
pub mod octree;
#[cfg(feature = "simd")]
pub mod rayon_bh;
pub mod rmn_soa;
pub mod sidm;

#[cfg(feature = "gpu")]
pub mod wgpu_monopole_bh;

pub use barnes_hut::BarnesHutGravity;
pub use let_tree::{
    DEFAULT_LEAF_MAX, LetTree, let_tree_prof_begin, let_tree_prof_end, let_tree_tile_prof_read,
};
#[cfg(feature = "simd")]
pub use octree::accel_from_let_soa;
pub use octree::{
    NO_CHILD, OctNode, Octree, RMN_FLOATS, RemoteMultipoleNode, WalkStats, accel_from_let,
    pack_let_nodes, unpack_let_nodes, walk_stats_begin, walk_stats_end,
};
#[cfg(feature = "simd")]
pub use rayon_bh::RayonBarnesHutGravity;
pub use rmn_soa::RmnSoa;
pub use sidm::{SidmParams, apply_sidm_scattering, scatter_probability};
#[cfg(feature = "gpu")]
pub use wgpu_monopole_bh::{WgpuBarnesHutGpu, WgpuMonopoleBarnesHut};
