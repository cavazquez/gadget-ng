//! Octree y Barnes-Hut para simulaciones N-body en **gadget-ng**.
pub mod barnes_hut;
pub mod let_tree;
pub mod octree;
#[cfg(feature = "simd")]
pub mod rayon_bh;
pub mod rmn_soa;
pub mod sidm;

pub use barnes_hut::BarnesHutGravity;
pub use let_tree::{
    let_tree_prof_begin, let_tree_prof_end, let_tree_tile_prof_read, LetTree, DEFAULT_LEAF_MAX,
};
#[cfg(feature = "simd")]
pub use octree::accel_from_let_soa;
pub use octree::{
    accel_from_let, pack_let_nodes, unpack_let_nodes, walk_stats_begin, walk_stats_end, OctNode,
    Octree, RemoteMultipoleNode, WalkStats, NO_CHILD, RMN_FLOATS,
};
#[cfg(feature = "simd")]
pub use rayon_bh::RayonBarnesHutGravity;
pub use rmn_soa::RmnSoa;
pub use sidm::{apply_sidm_scattering, scatter_probability, SidmParams};
