use gadget_ng_core::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct LightconeConfig {
    pub observer: Vec3,
    pub r_min: f64,
    pub r_max: f64,
    pub pencil_beam_axis: Option<Vec3>,
    pub pencil_beam_cos_half_angle: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LightconeHit {
    pub particle_index: usize,
    pub position: Vec3,
    pub distance: f64,
}

#[inline]
fn normalize(v: Vec3) -> Vec3 {
    let n = v.norm().max(1e-300);
    v / n
}

pub fn detect_lightcone_crossings(
    prev_positions: &[Vec3],
    curr_positions: &[Vec3],
    cfg: LightconeConfig,
) -> Vec<LightconeHit> {
    let mut out = Vec::new();
    let axis = cfg.pencil_beam_axis.map(normalize);
    for (i, (&x0, &x1)) in prev_positions.iter().zip(curr_positions.iter()).enumerate() {
        let r0v = x0 - cfg.observer;
        let r1v = x1 - cfg.observer;
        let r0 = r0v.norm();
        let r1 = r1v.norm();
        let crosses = (r0 < cfg.r_min && r1 >= cfg.r_min)
            || (r0 <= cfg.r_max && r1 > cfg.r_max)
            || (r0 > cfg.r_min && r0 < cfg.r_max);
        if !crosses {
            continue;
        }
        if let Some(a) = axis {
            let mu = (normalize(r1v)).dot(a);
            if mu < cfg.pencil_beam_cos_half_angle {
                continue;
            }
        }
        out.push(LightconeHit {
            particle_index: i,
            position: x1,
            distance: r1,
        });
    }
    out
}

#[derive(Debug, Clone)]
pub struct LensingMap {
    pub nside: usize,
    pub kappa: Vec<f64>,
    pub gamma1: Vec<f64>,
    pub gamma2: Vec<f64>,
}

impl LensingMap {
    pub fn new(nside: usize) -> Self {
        let n = nside * nside;
        Self {
            nside,
            kappa: vec![0.0; n],
            gamma1: vec![0.0; n],
            gamma2: vec![0.0; n],
        }
    }
}

/// Pipeline Born simple: acumula masas de hits en una malla angular cartesiana.
pub fn accumulate_born_lensing(
    hits: &[LightconeHit],
    masses: &[f64],
    observer: Vec3,
    nside: usize,
) -> LensingMap {
    let mut map = LensingMap::new(nside);
    for h in hits {
        if h.particle_index >= masses.len() {
            continue;
        }
        let r = h.position - observer;
        let rn = normalize(r);
        let u = ((rn.x + 1.0) * 0.5 * nside as f64).clamp(0.0, (nside - 1) as f64) as usize;
        let v = ((rn.y + 1.0) * 0.5 * nside as f64).clamp(0.0, (nside - 1) as f64) as usize;
        let pix = v * nside + u;
        let w = masses[h.particle_index] / h.distance.max(1e-6);
        map.kappa[pix] += w;
        map.gamma1[pix] += w * (rn.x * rn.x - rn.y * rn.y);
        map.gamma2[pix] += w * (2.0 * rn.x * rn.y);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_shell_crossing() {
        let prev = vec![Vec3::new(0.1, 0.0, 0.0)];
        let curr = vec![Vec3::new(0.6, 0.0, 0.0)];
        let cfg = LightconeConfig {
            observer: Vec3::zero(),
            r_min: 0.2,
            r_max: 1.0,
            pencil_beam_axis: None,
            pencil_beam_cos_half_angle: -1.0,
        };
        let hits = detect_lightcone_crossings(&prev, &curr, cfg);
        assert_eq!(hits.len(), 1);
    }
}
