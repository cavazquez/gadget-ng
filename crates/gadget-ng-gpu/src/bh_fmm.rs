//! Barnes–Hut FMM en GPU (WGSL): órdenes multipolares **1–4** alineados con
//! [`gadget_ng_tree::Octree::walk_accel_multipole`] (hexadecapolo vía `hex_dt_patterns` + pesos STF).
//!
//! MAC geométrico y **MAC relativo** (estimación quad / mono) cuando `use_relative != 0`.

use gadget_ng_gpu_layout::BhFmmGpuNode;
use std::sync::Arc;

const BH_FMM_SHADER: &str = concat!(
    r#"
const MAX_STACK: u32 = 64u;
const EMPTY: u32 = 0xffffffffu;

struct Params {
    eps2: f32,
    g: f32,
    theta: f32,
    err_tol: f32,
    root: u32,
    n_nodes: u32,
    n_all: u32,
    n_query: u32,
    multipole_order: u32,
    use_relative: u32,
    softened_multipoles: u32,
    mac_softening: u32,
}

struct FmmGpuNode {
    com_mass: vec4<f32>,
    center_half: vec4<f32>,
    children: array<u32, 8>,
    particle_idx: u32,
    pad: array<u32, 7>,
    quad: array<f32, 6>,
    oct: array<f32, 7>,
    hex: array<f32, 15>,
}

@group(0) @binding(0) var<uniform> params_u: Params;
@group(0) @binding(1) var<storage, read> bh_nodes: array<FmmGpuNode>;
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

fn quad_accel(r: vec3<f32>, q: array<f32, 6>, g: f32) -> vec3<f32> {
    let r2 = r.x * r.x + r.y * r.y + r.z * r.z;
    if r2 < 1e-30 { return vec3<f32>(0.0, 0.0, 0.0); }
    let qxx = q[0]; let qxy = q[1]; let qxz = q[2];
    let qyy = q[3]; let qyz = q[4]; let qzz = q[5];
    let r_inv = inverseSqrt(r2);
    let r5_inv = r_inv * r_inv * r_inv * r_inv * r_inv;
    let r7_inv = r5_inv * r_inv * r_inv;
    let qr_x = qxx * r.x + qxy * r.y + qxz * r.z;
    let qr_y = qxy * r.x + qyy * r.y + qyz * r.z;
    let qr_z = qxz * r.x + qyz * r.y + qzz * r.z;
    let rqr = qr_x * r.x + qr_y * r.y + qr_z * r.z;
    let c1 = g * r5_inv;
    let c2 = g * 2.5 * rqr * r7_inv;
    return vec3<f32>(c1 * qr_x - c2 * r.x, c1 * qr_y - c2 * r.y, c1 * qr_z - c2 * r.z);
}

fn quad_accel_soft(r: vec3<f32>, q: array<f32, 6>, g: f32, eps2: f32) -> vec3<f32> {
    let r2 = r.x * r.x + r.y * r.y + r.z * r.z + eps2;
    if r2 < 1e-30 { return vec3<f32>(0.0, 0.0, 0.0); }
    let qxx = q[0]; let qxy = q[1]; let qxz = q[2];
    let qyy = q[3]; let qyz = q[4]; let qzz = q[5];
    let r_inv = inverseSqrt(r2);
    let r5_inv = r_inv * r_inv * r_inv * r_inv * r_inv;
    let r7_inv = r5_inv * r_inv * r_inv;
    let qr_x = qxx * r.x + qxy * r.y + qxz * r.z;
    let qr_y = qxy * r.x + qyy * r.y + qyz * r.z;
    let qr_z = qxz * r.x + qyz * r.y + qzz * r.z;
    let rqr = qr_x * r.x + qr_y * r.y + qr_z * r.z;
    let c1 = g * r5_inv;
    let c2 = g * 2.5 * rqr * r7_inv;
    return vec3<f32>(c1 * qr_x - c2 * r.x, c1 * qr_y - c2 * r.y, c1 * qr_z - c2 * r.z);
}

fn oct_accel(r: vec3<f32>, o: array<f32, 7>, g: f32) -> vec3<f32> {
    let r2 = r.x * r.x + r.y * r.y + r.z * r.z;
    if r2 < 1e-30 { return vec3<f32>(0.0, 0.0, 0.0); }
    let o_xxx = o[0]; let o_xxy = o[1]; let o_xxz = o[2];
    let o_xyy = o[3]; let o_xyz = o[4]; let o_yyy = o[5]; let o_yzz = o[6];
    let o_xzz = -(o_xxx + o_xyy);
    let o_yyz = -(o_xxy + o_yyy);
    let o_zzz = -(o_xxz - o_xxy - o_yyy);
    let rx = r.x; let ry = r.y; let rz = r.z;
    let orr_x = o_xxx * rx * rx + 2.0 * o_xxy * rx * ry + 2.0 * o_xxz * rx * rz
        + o_xyy * ry * ry + 2.0 * o_xyz * ry * rz + o_xzz * rz * rz;
    let orr_y = o_xxy * rx * rx + 2.0 * o_xyy * rx * ry + 2.0 * o_xyz * rx * rz
        + o_yyy * ry * ry + 2.0 * o_yyz * ry * rz + o_yzz * rz * rz;
    let orr_z = o_xxz * rx * rx + 2.0 * o_xyz * rx * ry + 2.0 * o_xzz * rx * rz
        + o_yyz * ry * ry + 2.0 * o_yzz * ry * rz + o_zzz * rz * rz;
    let orrr = o_xxx * rx * rx * rx + 3.0 * o_xxy * rx * rx * ry + 3.0 * o_xxz * rx * rx * rz
        + 3.0 * o_xyy * rx * ry * ry + 6.0 * o_xyz * rx * ry * rz + 3.0 * o_xzz * rx * rz * rz
        + o_yyy * ry * ry * ry + 3.0 * o_yyz * ry * ry * rz + 3.0 * o_yzz * ry * rz * rz
        + o_zzz * rz * rz * rz;
    let r_inv = inverseSqrt(r2);
    let r7_inv = r_inv * r_inv * r_inv * r_inv * r_inv * r_inv * r_inv;
    let r9_inv = r7_inv * r_inv * r_inv;
    let c1 = -g * 0.5 * r7_inv;
    let c2 = g * (7.0 / 6.0) * orrr * r9_inv;
    return vec3<f32>(c1 * orr_x + c2 * rx, c1 * orr_y + c2 * ry, c1 * orr_z + c2 * rz);
}

fn oct_accel_soft(r: vec3<f32>, o: array<f32, 7>, g: f32, eps2: f32) -> vec3<f32> {
    let r2 = r.x * r.x + r.y * r.y + r.z * r.z + eps2;
    if r2 < 1e-30 { return vec3<f32>(0.0, 0.0, 0.0); }
    let o_xxx = o[0]; let o_xxy = o[1]; let o_xxz = o[2];
    let o_xyy = o[3]; let o_xyz = o[4]; let o_yyy = o[5]; let o_yzz = o[6];
    let o_xzz = -(o_xxx + o_xyy);
    let o_yyz = -(o_xxy + o_yyy);
    let o_zzz = -(o_xxz - o_xxy - o_yyy);
    let rx = r.x; let ry = r.y; let rz = r.z;
    let orr_x = o_xxx * rx * rx + 2.0 * o_xxy * rx * ry + 2.0 * o_xxz * rx * rz
        + o_xyy * ry * ry + 2.0 * o_xyz * ry * rz + o_xzz * rz * rz;
    let orr_y = o_xxy * rx * rx + 2.0 * o_xyy * rx * ry + 2.0 * o_xyz * rx * rz
        + o_yyy * ry * ry + 2.0 * o_yyz * ry * rz + o_yzz * rz * rz;
    let orr_z = o_xxz * rx * rx + 2.0 * o_xyz * rx * ry + 2.0 * o_xzz * rx * rz
        + o_yyz * ry * ry + 2.0 * o_yzz * ry * rz + o_zzz * rz * rz;
    let orrr = o_xxx * rx * rx * rx + 3.0 * o_xxy * rx * rx * ry + 3.0 * o_xxz * rx * rx * rz
        + 3.0 * o_xyy * rx * ry * ry + 6.0 * o_xyz * rx * ry * rz + 3.0 * o_xzz * rx * rz * rz
        + o_yyy * ry * ry * ry + 3.0 * o_yyz * ry * ry * rz + 3.0 * o_yzz * ry * rz * rz
        + o_zzz * rz * rz * rz;
    let r_inv = inverseSqrt(r2);
    let r7_inv = pow(r_inv, 7.0);
    let r9_inv = pow(r_inv, 9.0);
    let c1 = -g * 0.5 * r7_inv;
    let c2 = g * (7.0 / 6.0) * orrr * r9_inv;
    return vec3<f32>(c1 * orr_x + c2 * rx, c1 * orr_y + c2 * ry, c1 * orr_z + c2 * rz);
}
"#,
    include_str!("hex_dt_generated.inc.wgsl"),
    r#"
@group(0) @binding(6) var<storage, read> hex_w_flat: array<f32>;
const L4_FACT: f32 = 0.041666667;
fn hex_accel_flat(ni: u32, r: vec3<f32>, g: f32, eps2: f32) -> vec3<f32> {
    let r2 = r.x * r.x + r.y * r.y + r.z * r.z + eps2;
    if (r2 < 1e-30) { return vec3<f32>(0.0, 0.0, 0.0); }
    let rx = r.x; let ry = r.y; let rz = r.z;
    var ax = 0.0; var ay = 0.0; var az = 0.0;
    for (var p = 0u; p < 15u; p = p + 1u) {
        let w = hex_w_flat[ni * 15u + p];
        ax = ax + eval_dt_x(p, rx, ry, rz, r2) * w;
        ay = ay + eval_dt_y(p, rx, ry, rz, r2) * w;
        az = az + eval_dt_z(p, rx, ry, rz, r2) * w;
    }
    return vec3<f32>(-g * L4_FACT * ax, -g * L4_FACT * ay, -g * L4_FACT * az);
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
        if params_u.use_relative != 0u {
            if !inside && d > 1e-30 {
                let d2 = d * d;
                let a_mono_mag = params_u.g * M / (d2 + params_u.eps2);
                let q = node.quad;
                let q_frob2 = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]+q[4]*q[4]+q[5]*q[5];
                let q_frob = sqrt(q_frob2);
                var quad_mag = 0.0;
                if params_u.mac_softening == 0u {
                    quad_mag = q_frob / (d2 * d2 * d);
                } else {
                    let s2 = d2 + params_u.eps2;
                    quad_mag = q_frob / (s2 * s2 * sqrt(s2));
                }
                mac_ok = a_mono_mag > 1e-30 && (quad_mag / a_mono_mag) < params_u.err_tol;
            }
        } else {
            if params_u.theta > 0.0 && !inside && d > 1e-30 {
                mac_ok = (s / d) < params_u.theta;
            }
        }

        if mac_ok {
            let r_com = pi - com;
            let a_mono = plummer_pair(pi, M, com, params_u.eps2, params_u.g);
            var a_total = a_mono;
            if params_u.multipole_order >= 2u {
                var aq = vec3<f32>(0.0);
                if params_u.softened_multipoles != 0u {
                    aq = quad_accel_soft(r_com, node.quad, params_u.g, params_u.eps2);
                } else {
                    aq = quad_accel(r_com, node.quad, params_u.g);
                }
                a_total = a_total + aq;
            }
            if params_u.multipole_order >= 3u {
                var ao = vec3<f32>(0.0);
                if params_u.softened_multipoles != 0u {
                    ao = oct_accel_soft(r_com, node.oct, params_u.g, params_u.eps2);
                } else {
                    ao = oct_accel(r_com, node.oct, params_u.g);
                }
                a_total = a_total + ao;
            }
            if params_u.multipole_order >= 4u {
                let ah = hex_accel_flat(ni, r_com, params_u.g, params_u.eps2);
                a_total = a_total + ah;
            }
            ax += a_total.x;
            ay += a_total.y;
            az += a_total.z;
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
"#
);

struct GpuBhFmmCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

unsafe impl Send for GpuBhFmmCtx {}
unsafe impl Sync for GpuBhFmmCtx {}

/// Parámetros del kernel FMM (órdenes 1–4 en GPU; hex vía buffer auxiliar de pesos STF).
#[derive(Clone, Copy, Debug)]
pub struct BhFmmKernelParams {
    pub eps2: f32,
    pub g: f32,
    pub theta: f32,
    pub err_tol: f32,
    /// 1 = monopolo, 2 = +cuadrupolo, 3 = +octupolo, 4 = +hexadecapolo.
    pub multipole_order: u32,
    pub use_relative_criterion: bool,
    pub softened_multipoles: bool,
    /// 0 = Bare, 1 = Consistent (solo afecta MAC relativo).
    pub mac_softening: u32,
}

/// Walk Barnes–Hut multipolar en GPU (hasta orden 4).
#[derive(Clone)]
pub struct GpuBarnesHutFmm {
    ctx: Arc<GpuBhFmmCtx>,
}

impl GpuBarnesHutFmm {
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
                label: Some("gadget-ng-gpu-bh-fmm"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bh_fmm_wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BH_FMM_SHADER)),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bh_fmm_bgl"),
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
                super::solver::bgl_entry(
                    6,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bh_fmm_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bh_fmm_cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            ctx: Arc::new(GpuBhFmmCtx {
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
        nodes: &[BhFmmGpuNode],
        root: u32,
        query_idx: &[u32],
        kp: BhFmmKernelParams,
    ) -> Vec<f32> {
        let n_query = query_idx.len() as u32;
        if n_query == 0 {
            return Vec::new();
        }
        assert!(kp.multipole_order >= 1 && kp.multipole_order <= 4);
        let n_all = masses_f32.len() as u32;
        let n_nodes = nodes.len() as u32;
        assert_eq!(positions_f32.len(), 3 * n_all as usize);

        let mut hex_w_flat: Vec<f32> = vec![0.0; (nodes.len() * 15).max(1)];
        if kp.multipole_order >= 4 {
            for (ni, node) in nodes.iter().enumerate() {
                let h64: [f64; 15] = node.hex.map(|x| x as f64);
                let w = gadget_ng_gpu_layout::hex_pattern_weights(&h64);
                for p in 0..15 {
                    hex_w_flat[ni * 15 + p] = w[p] as f32;
                }
            }
        }

        #[repr(C)]
        struct Params {
            eps2: f32,
            g: f32,
            theta: f32,
            err_tol: f32,
            root: u32,
            n_nodes: u32,
            n_all: u32,
            n_query: u32,
            multipole_order: u32,
            use_relative: u32,
            softened_multipoles: u32,
            mac_softening: u32,
        }
        let params = Params {
            eps2: kp.eps2,
            g: kp.g,
            theta: kp.theta,
            err_tol: kp.err_tol,
            root,
            n_nodes,
            n_all,
            n_query,
            multipole_order: kp.multipole_order,
            use_relative: u32::from(kp.use_relative_criterion),
            softened_multipoles: u32::from(kp.softened_multipoles),
            mac_softening: kp.mac_softening,
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
                label: Some("bh_fmm_params"),
                contents: &params_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let nodes_bytes = unsafe {
            std::slice::from_raw_parts(
                nodes.as_ptr() as *const u8,
                nodes.len() * std::mem::size_of::<BhFmmGpuNode>(),
            )
        };
        let buf_nodes = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_fmm_nodes"),
                contents: nodes_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_pos = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_fmm_pos"),
                contents: &f32s_to_bytes(positions_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_mass = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_fmm_mass"),
                contents: &f32s_to_bytes(masses_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_idx = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_fmm_qidx"),
                contents: &u32s_to_bytes(query_idx),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_hex_w = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bh_fmm_hex_w"),
                contents: &f32s_to_bytes(&hex_w_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_bytes = 3 * n_query as u64 * 4;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bh_fmm_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bh_fmm_readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bh_fmm_bg"),
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_hex_w.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bh_fmm_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bh_fmm_pass"),
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
