use gadget_ng_core::{Particle, Vec3};

// ── Empaquetado de halos (8 f64 por partícula) ────────────────────────────────

/// Empaqueta partículas para halo exchange: `[global_id_bits, mass, x, y, z, vx, vy, vz]`.
pub fn pack_halo(particles: &[Particle]) -> Vec<f64> {
    let mut v = Vec::with_capacity(particles.len() * 8);
    for p in particles {
        v.push(f64::from_bits(p.global_id as u64));
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

/// Desempaqueta el buffer generado por [`pack_halo`].
pub fn unpack_halo(buf: &[f64]) -> Vec<Particle> {
    buf.chunks_exact(8)
        .map(|c| {
            Particle::new(
                f64::to_bits(c[0]) as usize,
                c[1],
                Vec3::new(c[2], c[3], c[4]),
                Vec3::new(c[5], c[6], c[7]),
            )
        })
        .collect()
}

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
            ptype: gadget_ng_core::ParticleType::DarkMatter,
            internal_energy: 0.0,
            smoothing_length: 0.0,
            metallicity: 0.0,
            stellar_age: 0.0,
            u_cold: 0.0,
            cr_energy: 0.0,
            h2_fraction: 0.0,
            b_field: gadget_ng_core::Vec3::zero(),
            psi_div: 0.0,
            dust_to_gas: 0.0,
            t_electron: 0.0,
        });
    }
    let mut particles = Vec::with_capacity(total_count);
    for slot in by_gid {
        particles.push(slot.expect("partícula faltante en gather global"));
    }
    particles
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    #[test]
    fn pack_unpack_halo_roundtrip() {
        let p0 = Particle::new(7, 2.5, Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3));
        let buf = pack_halo(&[p0]);
        let out = unpack_halo(&buf);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].global_id, 7);
        assert!((out[0].mass - 2.5).abs() < 1e-12);
        assert!((out[0].position.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pack_pm_unpack_flat_scatters_by_gid() {
        let local = vec![
            Particle::new(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero()),
            Particle::new(1, 2.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero()),
        ];
        let flat = pack_pm(&local);
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        unpack_pm_flat(&flat, &[10, 0], &mut pos, &mut mass, 2);
        assert_eq!(pos.len(), 2);
        assert!((mass[0] - 1.0).abs() < 1e-12);
        assert!((pos[1].x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pack_full_roundtrip_by_global_id() {
        let p = Particle::new(0, 4.0, Vec3::new(5.0, 6.0, 7.0), Vec3::new(8.0, 9.0, 10.0));
        let flat = pack_full(&[p]);
        let out = unpack_full_to_particles(&flat, 1);
        assert_eq!(out[0].mass, 4.0);
        assert!((out[0].position.y - 6.0).abs() < 1e-12);
    }
}
