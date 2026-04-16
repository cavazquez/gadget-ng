use gadget_ng_core::{Particle, Vec3};

/// Un paso leapfrog en forma kick–drift–kick (KDK) con paso `dt` fijo.
/// `compute` debe escribir aceleraciones alineadas con `particles` (mismo orden).
pub fn leapfrog_kdk_step(
    particles: &mut [Particle],
    dt: f64,
    scratch_acc: &mut [Vec3],
    mut compute: impl FnMut(&[Particle], &mut [Vec3]),
) {
    assert_eq!(particles.len(), scratch_acc.len());
    compute(particles, scratch_acc);
    for (p, &a) in particles.iter_mut().zip(scratch_acc.iter()) {
        p.velocity += a * (0.5 * dt);
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * dt;
    }
    compute(particles, scratch_acc);
    for (p, &a) in particles.iter_mut().zip(scratch_acc.iter()) {
        p.velocity += a * (0.5 * dt);
        p.acceleration = a;
    }
}

/// Factores de drift/kick cosmológicos para un paso KDK.
///
/// En el esquema KDK con momentum canónico (`p = a² dx_c/dt`):
/// 1. half-kick: `Δp = F · kick_half`
/// 2. drift:     `Δx_c = p · drift`
/// 3. half-kick: `Δp = F · kick_half2`
///
/// Los factores son integrales del factor de escala:
/// - `drift      = ∫_{t}^{t+dt}   dt'/a²(t')`
/// - `kick_half  = ∫_{t}^{t+dt/2} dt'/a(t')`
/// - `kick_half2 = ∫_{t+dt/2}^{t+dt} dt'/a(t')`
///
/// En ausencia de cosmología, usar [`CosmoFactors::flat`] para recuperar
/// el comportamiento Newtoniano (`drift = dt`, `kick_* = dt/2`).
#[derive(Debug, Clone, Copy)]
pub struct CosmoFactors {
    /// `∫_{t}^{t+dt} dt'/a²(t')` — factor de drift de posición.
    pub drift: f64,
    /// `∫_{t}^{t+dt/2} dt'/a(t')` — primer half-kick.
    pub kick_half: f64,
    /// `∫_{t+dt/2}^{t+dt} dt'/a(t')` — segundo half-kick.
    pub kick_half2: f64,
}

impl CosmoFactors {
    /// Factores planos sin cosmología: equivalentes a integración Newtoniana con `dt`.
    ///
    /// `drift = dt`, `kick_half = kick_half2 = dt/2`.
    #[inline]
    pub fn flat(dt: f64) -> Self {
        Self { drift: dt, kick_half: dt * 0.5, kick_half2: dt * 0.5 }
    }
}

/// Paso leapfrog KDK con factores de drift/kick cosmológicos.
///
/// Equivalente a [`leapfrog_kdk_step`] pero usa [`CosmoFactors`] en lugar
/// de `dt` plano. Pasa `CosmoFactors::flat(dt)` para comportamiento idéntico
/// al integrador Newtoniano estándar.
///
/// Las velocidades almacenadas en `particles` se interpretan como momentum
/// canónico `p = a² dx_c/dt`; las posiciones son coordenadas comóviles `x_c`.
pub fn leapfrog_cosmo_kdk_step(
    particles: &mut [Particle],
    cf: CosmoFactors,
    scratch_acc: &mut [Vec3],
    mut compute: impl FnMut(&[Particle], &mut [Vec3]),
) {
    assert_eq!(particles.len(), scratch_acc.len());
    compute(particles, scratch_acc);
    for (p, &a) in particles.iter_mut().zip(scratch_acc.iter()) {
        p.velocity += a * cf.kick_half;
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * cf.drift;
    }
    compute(particles, scratch_acc);
    for (p, &a) in particles.iter_mut().zip(scratch_acc.iter()) {
        p.velocity += a * cf.kick_half2;
        p.acceleration = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::cosmology::CosmologyParams;
    use gadget_ng_core::{Particle, Vec3};

    /// Verifica que `leapfrog_cosmo_kdk_step` con `CosmoFactors::flat(dt)`
    /// produce resultados bit-a-bit idénticos a `leapfrog_kdk_step` con `dt`.
    #[test]
    fn cosmo_step_flat_equals_newtonian() {
        let make_particle = || {
            let mut p = Particle::new(0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0));
            p.acceleration = Vec3::zero();
            p
        };

        let dt = 0.05_f64;
        let mut p1 = vec![make_particle()];
        let mut p2 = vec![make_particle()];
        let mut acc1 = vec![Vec3::zero()];
        let mut acc2 = vec![Vec3::zero()];

        let force = |ps: &[Particle], out: &mut [Vec3]| {
            out[0] = -ps[0].position; // oscilador armónico
        };

        leapfrog_kdk_step(&mut p1, dt, &mut acc1, force);
        leapfrog_cosmo_kdk_step(&mut p2, CosmoFactors::flat(dt), &mut acc2, force);

        assert_eq!(p1[0].position, p2[0].position);
        assert_eq!(p1[0].velocity, p2[0].velocity);
    }

    /// En un universo EdS con H₀→0 los factores cosmológicos se acercan a `dt` plano.
    /// Verifica que la diferencia en energía entre el modo cosmológico y el plano
    /// es pequeña cuando H₀ es pequeño.
    #[test]
    fn cosmo_step_eds_small_h0_close_to_flat() {
        let h0 = 1e-4_f64; // H₀ muy pequeño → casi Newtoniano
        let a0 = 1.0_f64;
        let dt = 0.01_f64;
        let p_cosmo = CosmologyParams::new(1.0, 0.0, h0);

        let (drift, kh, kh2) = p_cosmo.drift_kick_factors(a0, dt);
        let cf = CosmoFactors { drift, kick_half: kh, kick_half2: kh2 };
        let cf_flat = CosmoFactors::flat(dt);

        // La diferencia relativa en los factores debe ser O(H₀·dt) ≈ 1e-6.
        let tol = 1e-3 * dt;
        assert!((cf.drift - cf_flat.drift).abs() < tol);
        assert!((cf.kick_half - cf_flat.kick_half).abs() < tol);
    }
}
