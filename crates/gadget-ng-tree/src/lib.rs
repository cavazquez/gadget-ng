//! Octree y Barnes-Hut para simulaciones N-body en **gadget-ng**.
pub mod barnes_hut;
pub mod let_tree;
pub mod octree;
pub mod rmn_soa;
#[cfg(feature = "simd")]
pub mod rayon_bh;

pub use barnes_hut::BarnesHutGravity;
pub use let_tree::{let_tree_prof_begin, let_tree_prof_end, LetTree, DEFAULT_LEAF_MAX};
pub use octree::{
    accel_from_let, pack_let_nodes, unpack_let_nodes, walk_stats_begin, walk_stats_end, OctNode,
    Octree, RemoteMultipoleNode, WalkStats, RMN_FLOATS, NO_CHILD,
};
pub use rmn_soa::RmnSoa;
#[cfg(feature = "simd")]
pub use octree::accel_from_let_soa;
#[cfg(feature = "simd")]
pub use rayon_bh::RayonBarnesHutGravity;
