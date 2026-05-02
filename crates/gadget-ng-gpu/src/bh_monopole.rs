//! Barnes–Hut **monopolo** en GPU (WGSL): MAC geométrico coherente con
//! `Octree::walk_accel_multipole` cuando `multipole_order = 1`.
//!
//! Multipolos superiores y criterio relativo **no** están en este kernel (pendiente).

use gadget_ng_gpu_layout::BhMonopoleGpuNode;
use std::sync::Arc;

const BH_MONO_SHADER: &str = r#"
const MAX_STACK: u32 = 64u;
const EMPTY: u32 = 0xffffffffu;

struct Params {
    eps2: f32,
    g: f32,
    theta: f32,
    _pad: f32,
    root: u32,
    n_nodes: u32,
    n_all: u32,
    n_query: u32,
}

struct GpuNode {
    com_mass: vec4<f32>,
    center_half: vec4<f32>,
    children: array<u32, 8>,
    particle_idx: u32,
    pad: array<u32, 7>,
}

@group(0) @binding(0) var<uniform> params_u: Params;
@group(0) @binding(1) var<storage, read> bh_nodes: array<GpuNode>;
@group(0) @binding(2) var<storage, read> positions: array<f32>;
@group(0) @binding(3) var<storage, read> masses: array<f32>;
@group(0) @binding(4) var<storage, read> query_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_accs: array<f32>;

fn plummer_pair(pi: vec3<f32>, mj: f32, pj: vec3<f32>, eps2: f32, g: f32) -> vec3<f32> {
    let dx = pj.x - pi.x;
    let dy = pj.y - pi.y;
    let dz = pj.z - pi.z;
    let r2 = dx * dx + dy * dy + dz * dz + eps2;
    let inv = inverseSqrt(r2);
    let r3inv = inv * inv * inv;
    return vec3<f32>(dx, dy, dz) * (g * mj * r3inv);
}

fn inside_cell(pi: vec3<f32>, center: vec3<f32>, half: f32) -> bool {
    let tol = 1e-14 * (1.0 + half);
    return abs(pi.x - center.x) <= half + tol
        && abs(pi.y - center.y) <= half + tol
        && abs(pi.z - center.z) <= half + tol;
}

fn is_leaf_empty_children(c: array<u32, 8>) -> bool {
    return c[0] == EMPTY && c[1] == EMPTY && c[2] == EMPTY && c[3] == EMPTY
        && c[4] == EMPTY && c[5] == EMPTY && c[6] == EMPTY && c[7] == EMPTY;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if qi >= params_u.n_query { return; }

    let gi = query_indices[qi];
    let px = positions[3u * gi];
    let py = positions[3u * gi + 1u];
    let pz = positions[3u * gi + 2u];
    let pi = vec3<f32>(px, py, pz);

    var ax = 0.0;
    var ay = 0.0;
    var az = 0.0;

    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = params_u.root;

    loop {
        if sp == 0u { break; }
        sp = sp - 1u;
        let ni = stack[sp];
        let node = bh_nodes[ni];
        let M = node.com_mass.w;
        if M <= 0.0 { continue; }

        let leaf = is_leaf_empty_children(node.children);

        if leaf {
            let pj_idx = node.particle_idx;
            if pj_idx != EMPTY && pj_idx != gi {
                let j = pj_idx;
                let jx = positions[3u * j];
                let jy = positions[3u * j + 1u];
                let jz = positions[3u * j + 2u];
                let pj = vec3<f32>(jx, jy, jz);
                let mj = masses[j];
                let a = plummer_pair(pi, mj, pj, params_u.eps2, params_u.g);
                ax += a.x;
                ay += a.y;
                az += a.z;
            }
            continue;
        }

        let com = vec3<f32>(node.com_mass.x, node.com_mass.y, node.com_mass.z);
        let center = vec3<f32>(node.center_half.x, node.center_half.y, node.center_half.z);
        let half = node.center_half.w;
        let s = 2.0 * half;
        let rvec = pi - com;
        let d = length(rvec);
        let inside = inside_cell(pi, center, half);
        var mac_ok = false;
        if params_u.theta > 0.0 && !inside && d > 1e-30 {
            mac_ok = (s / d) < params_u.theta;
        }

        if mac_ok {
            let a = plummer_pair(pi, M, com, params_u.eps2, params_u.g);
            ax += a.x;
            ay += a.y;
            az += a.z;
            continue;
        }

        for (var k = 0u; k < 8u; k++) {
            let cid = node.children[k];
            if cid != EMPTY && sp < MAX_STACK - 1u {
                stack[sp] = cid;
                sp = sp + 1u;
            }
        }
    }

    out_accs[3u * qi] = ax;
    out_accs[3u * qi + 1u] = ay;
    out_accs[3u * qi + 2u] = az;
}
"#;

struct GpuBhCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

unsafe impl Send for GpuBhCtx {}
unsafe impl Sync for GpuBhCtx {}

/// Recorrido Barnes–Hut monopolo en GPU (árbol ya construido en CPU y exportado).
#[derive(Clone)]
pub struct GpuBarnesHutMonopole {
    ctx: Arc<GpuBhCtx>,
}

impl GpuBarnesHutMonopole {
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
                label: Some("gadget-ng-gpu-bh"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bh_monopole_wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BH_MONO_SHADER)),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bh_mono_bgl"),
            entries: &[
                super::solver::bgl_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                super::solver::bgl_entry(
                    1,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                super::solver::bgl_entry(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                super::solver::bgl_entry(
                    3,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                super::solver::bgl_entry(
                    4,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                super::solver::bgl_entry(
                    5,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bh_mono_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bh_mono_cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            ctx: Arc::new(GpuBhCtx {
                device,
                queue,
                pipeline,
                bgl,
            }),
        })
    }

    /// `positions_f32`: `3 * n_all`, `nodes`: exportación del octree, `root` índice raíz.
    pub fn compute_accelerations_raw(
        &self,
        positions_f32: &[f32],
        masses_f32: &[f32],
        nodes: &[BhMonopoleGpuNode],
        root: u32,
        query_idx: &[u32],
        eps2: f32,
        g: f32,
        theta: f32,
    ) -> Vec<f32> {
        let n_query = query_idx.len() as u32;
        if n_query == 0 {
            return Vec::new();
        }
        let n_all = masses_f32.len() as u32;
        let n_nodes = nodes.len() as u32;
        assert_eq!(positions_f32.len(), 3 * n_all as usize);

        let ctx = &*self.ctx;
        use wgpu::util::DeviceExt;

        #[repr(C)]
        struct Params {
            eps2: f32,
            g: f32,
            theta: f32,
            _pad: f32,
            root: u32,
            n_nodes: u32,
            n_all: u32,
            n_query: u32,
        }
        let params = Params {
            eps2,
            g,
            theta,
            _pad: 0.0,
            root,
            n_nodes,
            n_all,
            n_query,
        };
        let params_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
            .to_vec()
        };

        let buf_params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_params"),
                contents: &params_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let nodes_bytes = unsafe {
            std::slice::from_raw_parts(
                nodes.as_ptr() as *const u8,
                nodes.len() * std::mem::size_of::<BhMonopoleGpuNode>(),
            )
        };
        let buf_nodes = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_nodes"),
                contents: nodes_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_pos = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_pos"),
                contents: &f32s_to_bytes(positions_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_mass = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_mass"),
                contents: &f32s_to_bytes(masses_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_idx = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_qidx"),
                contents: &u32s_to_bytes(query_idx),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_bytes = 3 * n_query as u64 * 4;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bh_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bh_readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bh_bg"),
            layout: &ctx.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bh_mono_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bh_mono_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let workgroups = n_query.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_rb, 0, out_bytes);
        ctx.queue.submit(Some(encoder.finish()));

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
