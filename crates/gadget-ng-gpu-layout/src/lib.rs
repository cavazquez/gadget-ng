//! Layout binario compartido CPU ↔ GPU para Barnes–Hut **monopolo** (wgpu/CUDA).
//!
//! Crate mínimo independiente de `gadget-ng-core` y `gadget-ng-gpu` para evitar
//! ciclos de paquetes (core ↔ gpu con feature opcional).

/// Índice de partícula inválido / hoja sin un solo cuerpo indexado (equivale a `Option::None`).
pub const BH_GPU_NO_PARTICLE: u32 = u32::MAX;

/// Nodo octree para kernel Barnes–Hut monopolo en GPU (96 bytes, `repr(C)`).
///
/// Debe coincidir con el struct `GpuNode` en `gadget-ng-gpu` / WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhMonopoleGpuNode {
    pub com: [f32; 3],
    pub mass: f32,
    pub center: [f32; 3],
    pub half: f32,
    pub children: [u32; 8],
    pub particle_idx: u32,
    pub _reserved: [u32; 7],
}

/// Nodo FMM para WGSL: monopolo + tensores STF (órdenes 2–4) en **f32**.
///
/// Debe coincidir con el struct `FmmGpuNode` en el shader WGSL (`gadget-ng-gpu`, `bh_fmm`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhFmmGpuNode {
    pub com: [f32; 3],
    pub mass: f32,
    pub center: [f32; 3],
    pub half: f32,
    pub children: [u32; 8],
    pub particle_idx: u32,
    pub _reserved: [u32; 7],
    pub quad: [f32; 6],
    pub oct: [f32; 7],
    pub hex: [f32; 15],
}
