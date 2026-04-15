use gadget_ng_core::{Particle, Vec3};

/// Estado mínimo para gravedad: `global_id`, `mass`, `px`, `py`, `pz`.
pub fn pack_pm(local: &[Particle]) -> Vec<f64> {
    let mut v = Vec::with_capacity(local.len() * 5);
    for p in local {
        v.push(p.global_id as f64);
        v.push(p.mass);
        v.push(p.position.x);
        v.push(p.position.y);
        v.push(p.position.z);
    }
    v
}

/// `recv_counts[i]` = número de `f64` del rango `i` (múltiplo de 5).
pub fn unpack_pm_flat(
    flat: &[f64],
    recv_counts: &[i32],
    global_positions: &mut Vec<Vec3>,
    global_masses: &mut Vec<f64>,
    total_count: usize,
) {
    global_positions.clear();
    global_masses.clear();
    global_positions.resize(total_count, Vec3::zero());
    global_masses.resize(total_count, 0.0);
    let mut off = 0usize;
    for &c in recv_counts {
        let nf = c as usize;
        assert_eq!(nf % 5, 0);
        for chunk in flat[off..off + nf].chunks(5) {
            let gid = chunk[0] as usize;
            let m = chunk[1];
            let x = chunk[2];
            let y = chunk[3];
            let z = chunk[4];
            if gid < total_count {
                global_masses[gid] = m;
                global_positions[gid] = Vec3::new(x, y, z);
            }
        }
        off += nf;
    }
    debug_assert_eq!(off, flat.len());
}

/// Snapshot: `global_id`, `mass`, posición, velocidad (8 `f64` por partícula).
pub fn pack_full(local: &[Particle]) -> Vec<f64> {
    let mut v = Vec::with_capacity(local.len() * 8);
    for p in local {
        v.push(p.global_id as f64);
        v.push(p.mass);
        v.push(p.position.x);
        v.push(p.position.y);
        v.push(p.position.z);
        v.push(p.velocity.x);
        v.push(p.velocity.y);
        v.push(p.velocity.z);
    }
    v
}

pub fn unpack_full_to_particles(flat: &[f64], total_count: usize) -> Vec<Particle> {
    let mut by_gid: Vec<Option<Particle>> = (0..total_count).map(|_| None).collect();
    for ch in flat.chunks(8) {
        if ch.len() < 8 {
            break;
        }
        let gid = ch[0] as usize;
        if gid >= total_count {
            continue;
        }
        let mass = ch[1];
        let position = Vec3::new(ch[2], ch[3], ch[4]);
        let velocity = Vec3::new(ch[5], ch[6], ch[7]);
        by_gid[gid] = Some(Particle {
            global_id: gid,
            mass,
            position,
            velocity,
            acceleration: Vec3::zero(),
        });
    }
    let mut particles = Vec::with_capacity(total_count);
    for slot in by_gid {
        particles.push(slot.expect("partícula faltante en gather global"));
    }
    particles
}
