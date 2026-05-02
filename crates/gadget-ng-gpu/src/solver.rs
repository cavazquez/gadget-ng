//! Solver GPU real — gravedad directa Plummer con wgpu (WGSL compute shader).
//!
//! ## Precisión
//!
//! Los cálculos internos en el shader usan **f32**. Las posiciones, masas y
//! aceleraciones f64 del motor se convierten a f32 antes de la transferencia al
//! device y de vuelta a f64 al recibirlos. El error relativo introducido es
//! O(ε_machine_f32) ≈ 1e-7, aceptable para la mayoría de simulaciones N-body.
//!
//! ## Degradación elegante
//!
//! [`GpuDirectGravity::try_new`] devuelve `None` si no hay adaptador GPU
//! disponible (headless, CI, máquina sin GPU). El motor puede caer en CPU.

use std::sync::Arc;

// ── Shader WGSL ──────────────────────────────────────────────────────────────

const SHADER_SRC: &str = r#"
struct Params {
    eps2    : f32,
    g       : f32,
    n_all   : u32,
    n_query : u32,
};

@group(0) @binding(0) var<uniform>              params        : Params;
@group(0) @binding(1) var<storage, read>        positions     : array<f32>; // 3 × n_all
@group(0) @binding(2) var<storage, read>        masses        : array<f32>; // n_all
@group(0) @binding(3) var<storage, read>        query_indices : array<u32>; // n_query
@group(0) @binding(4) var<storage, read_write>  out_accs      : array<f32>; // 3 × n_query

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if qi >= params.n_query { return; }

    let gi = query_indices[qi];
    let px = positions[3u * gi];
    let py = positions[3u * gi + 1u];
    let pz = positions[3u * gi + 2u];

    var ax = 0.0f;
    var ay = 0.0f;
    var az = 0.0f;

    for (var j = 0u; j < params.n_all; j++) {
        if j == gi { continue; }
        let dx = positions[3u * j]      - px;
        let dy = positions[3u * j + 1u] - py;
        let dz = positions[3u * j + 2u] - pz;
        let r2soft  = dx * dx + dy * dy + dz * dz + params.eps2;
        let r3inv   = inverseSqrt(r2soft) / r2soft;
        let mj      = masses[j];
        ax += mj * dx * r3inv;
        ay += mj * dy * r3inv;
        az += mj * dz * r3inv;
    }

    out_accs[3u * qi]      = params.g * ax;
    out_accs[3u * qi + 1u] = params.g * ay;
    out_accs[3u * qi + 2u] = params.g * az;
}
"#;

// ── Contexto GPU ─────────────────────────────────────────────────────────────

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

// SAFETY: wgpu::Device, Queue, ComputePipeline y BindGroupLayout son Send+Sync.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

// ── GpuDirectGravity ─────────────────────────────────────────────────────────

/// Solver de gravedad directa GPU usando wgpu (WGSL compute shader).
///
/// Construir con [`GpuDirectGravity::try_new`]; devuelve `None` si no hay GPU.
/// La impl de `GravitySolver` vive en `gadget_ng_core::gpu_bridge` para evitar
/// la dependencia circular `gadget-ng-gpu ↔ gadget-ng-core`.
#[derive(Clone)]
pub struct GpuDirectGravity {
    ctx: Arc<GpuContext>,
}

impl std::fmt::Debug for GpuDirectGravity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuDirectGravity { wgpu }")
    }
}

impl GpuDirectGravity {
    /// Intenta inicializar el contexto wgpu y compilar el shader.
    ///
    /// Devuelve `None` si no hay adaptador GPU disponible.
    pub fn try_new() -> Option<Self> {
        pollster::block_on(Self::try_new_async())
    }

    async fn try_new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gadget-ng-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("direct_gravity_wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SRC)),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("direct_gravity_bgl"),
            entries: &[
                bgl_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(
                    1,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(
                    3,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                bgl_entry(
                    4,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("direct_gravity_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("direct_gravity_cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            ctx: Arc::new(GpuContext {
                device,
                queue,
                pipeline,
                bgl,
            }),
        })
    }

    /// Lanza el compute shader de gravedad directa.
    ///
    /// # Argumentos
    /// - `positions_f32` — `3 × n_all` floats `[x0,y0,z0, x1,y1,z1, …]`
    /// - `masses_f32`    — `n_all` masas
    /// - `query_idx`     — índices globales de las `n_query` partículas a evaluar
    /// - `eps2`, `g`     — suavizado Plummer² y constante gravitatoria (f32)
    ///
    /// Devuelve `3 × n_query` aceleraciones `[ax0,ay0,az0, …]` en f32.
    pub fn compute_accelerations_raw(
        &self,
        positions_f32: &[f32],
        masses_f32: &[f32],
        query_idx: &[u32],
        eps2: f32,
        g: f32,
    ) -> Vec<f32> {
        let n_query = query_idx.len() as u32;
        if n_query == 0 {
            return Vec::new();
        }
        let n_all = masses_f32.len() as u32;
        let ctx = &*self.ctx;

        use wgpu::util::DeviceExt;

        // ── Uniform: [eps2_bits, g_bits, n_all, n_query] como 4 × u32 ────────
        let params_bytes: Vec<u8> = [eps2.to_bits(), g.to_bits(), n_all, n_query]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let buf_params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &params_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let buf_pos = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("positions"),
                contents: &f32s_to_bytes(positions_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_mass = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("masses"),
                contents: &f32s_to_bytes(masses_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_idx = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("query_idx"),
                contents: &u32s_to_bytes(query_idx),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_bytes = 3 * n_query as u64 * 4; // 4 bytes por f32
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_accs"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &ctx.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gravity_enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gravity_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let workgroups = n_query.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&buf_out, 0, &buf_rb, 0, out_bytes);
        ctx.queue.submit(Some(enc.finish()));

        // Readback síncrono
        let (tx, rx) = std::sync::mpsc::channel();
        buf_rb
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU device lost during poll");
        rx.recv()
            .expect("map_async recv")
            .expect("GPU buffer map failed");

        let view = buf_rb.slice(..).get_mapped_range();
        let result = bytes_to_f32s(&view);
        drop(view);
        buf_rb.unmap();
        result
    }
}

// ── Utilidades internas ───────────────────────────────────────────────────────

pub(crate) fn bgl_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|u| u.to_le_bytes()).collect()
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Gravedad directa CPU (referencia) con suavizado Plummer — para comparar.
    fn direct_gravity_cpu(
        positions: &[[f32; 3]],
        masses: &[f32],
        query_idx: &[u32],
        eps2: f32,
        g: f32,
    ) -> Vec<[f32; 3]> {
        query_idx
            .iter()
            .map(|&qi| {
                let [px, py, pz] = positions[qi as usize];
                let (mut ax, mut ay, mut az) = (0.0f32, 0.0, 0.0);
                for (j, &[jx, jy, jz]) in positions.iter().enumerate() {
                    if j == qi as usize {
                        continue;
                    }
                    let dx = jx - px;
                    let dy = jy - py;
                    let dz = jz - pz;
                    let r2soft = dx * dx + dy * dy + dz * dz + eps2;
                    let r3inv = r2soft.sqrt().recip() / r2soft;
                    ax += masses[j] * dx * r3inv;
                    ay += masses[j] * dy * r3inv;
                    az += masses[j] * dz * r3inv;
                }
                [g * ax, g * ay, g * az]
            })
            .collect()
    }

    #[test]
    fn gpu_matches_cpu_two_particles() {
        let Some(gpu) = GpuDirectGravity::try_new() else {
            eprintln!("SKIP gpu_matches_cpu_two_particles: no GPU disponible");
            return;
        };
        let positions = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let masses = [1.0f32, 1.0];
        let flat_pos: Vec<f32> = positions.iter().flat_map(|p| *p).collect();
        let eps2 = 0.01f32;
        let g = 1.0f32;
        let idx = [0u32, 1];
        let raw = gpu.compute_accelerations_raw(&flat_pos, &masses, &idx, eps2, g);
        let cpu = direct_gravity_cpu(&positions, &masses, &idx, eps2, g);
        for (k, cpu_acc) in cpu.iter().enumerate() {
            for comp in 0..3 {
                let diff = (raw[3 * k + comp] - cpu_acc[comp]).abs();
                assert!(
                    diff < 1e-5,
                    "k={k} comp={comp}: GPU={} CPU={} diff={diff}",
                    raw[3 * k + comp],
                    cpu_acc[comp]
                );
            }
        }
    }

    #[test]
    fn gpu_matches_cpu_random_particles() {
        let Some(gpu) = GpuDirectGravity::try_new() else {
            eprintln!("SKIP gpu_matches_cpu_random_particles: no GPU disponible");
            return;
        };
        // 16 partículas pseudo-aleatorias
        let n: usize = 16;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = i as f32 * 0.37;
                [t.sin(), t.cos(), (t * 0.5).sin()]
            })
            .collect();
        let masses: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let flat_pos: Vec<f32> = positions.iter().flat_map(|p| *p).collect();
        let idx: Vec<u32> = (0..n as u32).collect();
        let eps2 = 0.01f32;
        let g = 1.0f32;
        let raw = gpu.compute_accelerations_raw(&flat_pos, &masses, &idx, eps2, g);
        let cpu = direct_gravity_cpu(&positions, &masses, &idx, eps2, g);
        for (k, cpu_acc) in cpu.iter().enumerate() {
            for comp in 0..3 {
                let diff = (raw[3 * k + comp] - cpu_acc[comp]).abs();
                let rel = diff / (cpu_acc[comp].abs() + 1e-12);
                assert!(rel < 1e-4, "k={k} comp={comp}: rel_err={rel:.2e}");
            }
        }
    }

    #[test]
    fn gpu_empty_query_returns_empty() {
        let Some(gpu) = GpuDirectGravity::try_new() else {
            return;
        };
        let raw = gpu.compute_accelerations_raw(&[0.0, 0.0, 0.0], &[1.0], &[], 0.01, 1.0);
        assert!(raw.is_empty());
    }
}
