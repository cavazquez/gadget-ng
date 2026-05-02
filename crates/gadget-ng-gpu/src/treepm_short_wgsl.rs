//! Corto alcance **TreePM** en WGSL: mismo árbol exportado que Barnes–Hut,
//! recorrido alineado con `gadget_ng_treepm::short_range::walk_short_range` (aperiódico).
//!
//! Largo alcance PM sigue en CPU / CUDA; este módulo sólo acelera el término `erfc`.

use gadget_ng_gpu_layout::BhMonopoleGpuNode;
use std::sync::Arc;

const TREEPM_SR_SHADER: &str = r#"
const MAX_STACK: u32 = 64u;
const EMPTY: u32 = 0xffffffffu;

struct Params {
    eps2: f32,
    g: f32,
    r_split: f32,
    r_cut2: f32,
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

fn erfc_approx_pos(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return poly * exp(-x * x);
}

fn erfc_approx(x_in: f32) -> f32 {
    if x_in < 0.0 {
        return 2.0 - erfc_approx_pos(-x_in);
    }
    return erfc_approx_pos(x_in);
}

fn erfc_factor(r: f32, r_split: f32) -> f32 {
    let x = r / (1.4142135623730951 * r_split);
    return erfc_approx(x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if qi >= params_u.n_query { return; }

    let gi = query_indices[qi];
    let px = positions[3u * gi];
    let py = positions[3u * gi + 1u];
    let pz = positions[3u * gi + 2u];
    let xi = vec3<f32>(px, py, pz);

    var ax = 0.0;
    var ay = 0.0;
    var az = 0.0;

    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = params_u.root;

    let rc2 = params_u.r_cut2;

    loop {
        if sp == 0u { break; }
        sp = sp - 1u;
        let ni = stack[sp];
        let node = bh_nodes[ni];
        let M = node.com_mass.w;
        if M <= 0.0 { continue; }

        let dx_c = node.com_mass.x - xi.x;
        let dy_c = node.com_mass.y - xi.y;
        let dz_c = node.com_mass.z - xi.z;
        let h = node.center_half.w;
        let ex = max(abs(dx_c) - h, 0.0);
        let ey = max(abs(dy_c) - h, 0.0);
        let ez = max(abs(dz_c) - h, 0.0);
        if ex * ex + ey * ey + ez * ez > rc2 {
            continue;
        }

        let leaf = node.children[0] == EMPTY && node.children[1] == EMPTY
            && node.children[2] == EMPTY && node.children[3] == EMPTY
            && node.children[4] == EMPTY && node.children[5] == EMPTY
            && node.children[6] == EMPTY && node.children[7] == EMPTY;

        if leaf {
            let pj_idx = node.particle_idx;
            if pj_idx != EMPTY && pj_idx != gi {
                let j = pj_idx;
                let jx = positions[3u * j];
                let jy = positions[3u * j + 1u];
                let jz = positions[3u * j + 2u];
                let rx = jx - xi.x;
                let ry = jy - xi.y;
                let rz = jz - xi.z;
                let r2 = rx * rx + ry * ry + rz * rz + params_u.eps2;
                let r = sqrt(r2);
                let w = erfc_factor(r, params_u.r_split);
                let inv3 = params_u.g * masses[j] * w / (r2 * r);
                ax += rx * inv3;
                ay += ry * inv3;
                az += rz * inv3;
            }
            continue;
        }

        let d2 = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c;
        let r_cut = sqrt(rc2);
        let use_mono = h < 0.1 * r_cut && d2 > 1e-30;

        if use_mono {
            let r2 = d2 + params_u.eps2;
            let r = sqrt(r2);
            let w = erfc_factor(r, params_u.r_split);
            let inv3 = params_u.g * M * w / (r2 * r);
            ax += dx_c * inv3;
            ay += dy_c * inv3;
            az += dz_c * inv3;
        } else {
            for (var k = 0u; k < 8u; k++) {
                let cid = node.children[k];
                if cid != EMPTY && sp < MAX_STACK - 1u {
                    stack[sp] = cid;
                    sp = sp + 1u;
                }
            }
        }
    }

    out_accs[3u * qi] = ax;
    out_accs[3u * qi + 1u] = ay;
    out_accs[3u * qi + 2u] = az;
}
"#;

struct GpuTreePmSrCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

unsafe impl Send for GpuTreePmSrCtx {}
unsafe impl Sync for GpuTreePmSrCtx {}

/// Corto alcance TreePM (kernel erfc) en GPU; construir árbol en CPU y exportar con [`BhMonopoleGpuNode`].
#[derive(Clone)]
pub struct GpuTreePmShortRange {
    ctx: Arc<GpuTreePmSrCtx>,
}

impl GpuTreePmShortRange {
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
                label: Some("gadget-ng-gpu-treepm-sr"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("treepm_sr_wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TREEPM_SR_SHADER)),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("treepm_sr_bgl"),
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
            label: Some("treepm_sr_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("treepm_sr_cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            ctx: Arc::new(GpuTreePmSrCtx {
                device,
                queue,
                pipeline,
                bgl,
            }),
        })
    }

    pub fn compute_accelerations_raw(
        &self,
        positions_f32: &[f32],
        masses_f32: &[f32],
        nodes: &[BhMonopoleGpuNode],
        root: u32,
        query_idx: &[u32],
        eps2: f32,
        g: f32,
        r_split: f32,
        r_cut2: f32,
    ) -> Vec<f32> {
        let n_query = query_idx.len() as u32;
        if n_query == 0 {
            return Vec::new();
        }
        let n_all = masses_f32.len() as u32;
        let n_nodes = nodes.len() as u32;
        assert_eq!(positions_f32.len(), 3 * n_all as usize);

        #[repr(C)]
        struct Params {
            eps2: f32,
            g: f32,
            r_split: f32,
            r_cut2: f32,
            root: u32,
            n_nodes: u32,
            n_all: u32,
            n_query: u32,
        }
        let params = Params {
            eps2,
            g,
            r_split,
            r_cut2,
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

        let ctx = &*self.ctx;
        use wgpu::util::DeviceExt;

        let buf_params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tpm_sr_params"),
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
                label: Some("tpm_sr_nodes"),
                contents: nodes_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_pos = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tpm_sr_pos"),
                contents: &f32s_to_bytes(positions_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_mass = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tpm_sr_mass"),
                contents: &f32s_to_bytes(masses_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_idx = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tpm_sr_qidx"),
                contents: &u32s_to_bytes(query_idx),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_bytes = 3 * n_query as u64 * 4;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tpm_sr_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tpm_sr_readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tpm_sr_bg"),
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
                label: Some("tpm_sr_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tpm_sr_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_query.div_ceil(64), 1, 1);
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
