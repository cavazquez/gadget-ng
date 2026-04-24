# Plan de Validación — Tareas HPC Pendientes

**Fecha:** abril 2026  
**Estado del proyecto:** Phases 1–160 completadas (física completa).  
**Propósito:** Guía de implementación y validación autónoma para las tres tareas
HPC/ingeniería restantes. Diseñado para dejarse corriendo sin supervisión:
cada tarea incluye una suite de tests que sirven como criterio de aceptación.

---

## Índice

1. [V1 — GPU CUDA/HIP: kernels reales de N-cuerpos y PM](#v1)
2. [V2 — Block timesteps acoplados a árbol distribuido MPI](#v2)
3. [V3 — ICs MHD cosmológicas + validaciones cuantitativas](#v3)
4. [Checklist de ejecución autónoma](#checklist)

---

<a name="v1"></a>
## V1 — GPU CUDA/HIP: kernels reales de N-cuerpos y PM

### Estado actual

El crate `gadget-ng-gpu` tiene:
- `GpuDirectGravity` con kernel WGSL via **wgpu** (O(N²), f32, funcional).
- `GpuParticlesSoA`: layout SoA listo para device.
- Bridge `gadget_ng_core::gpu_bridge` que activa el solver vía `use_gpu = true`.

Pendiente: reemplazar el shader WGSL por kernels **CUDA** (`.cu`) o **HIP** (`.hip`)
para clusters con GPUs NVIDIA/AMD. El objetivo secundario es un solver PM en GPU
(FFT de P(k) en device, `cufft` / `hipfft`).

### Diseño de implementación

#### 1. Feature flags

```toml
# crates/gadget-ng-gpu/Cargo.toml
[features]
default = []
cuda  = ["dep:cuda-sys"]          # nvcc pathway
hip   = ["dep:hip-sys"]           # hipcc pathway
pm_gpu = ["cuda", "dep:cufft-sys"] # FFT en device
```

#### 2. Kernels a escribir

| Archivo | Función | Descripción |
|---------|---------|-------------|
| `kernels/direct_gravity.cu` | `__global__ direct_gravity_kernel` | Plummer O(N²) en f64 con shared memory tile |
| `kernels/direct_gravity.hip` | idem | traducción HIP (hipify automático o manual) |
| `kernels/pm_fft.cu` | `pm_forward_fft_kernel` | CUFFT R2C 3D para malla de densidad |
| `src/cuda_solver.rs` | `CudaDirectGravity` | wrapper Rust via `cuda-sys` |
| `src/hip_solver.rs` | `HipDirectGravity` | wrapper Rust via `hip-sys` |
| `src/pm_gpu.rs` | `PmGpuSolver` | pipeline CIC→FFT→convolución→IFFT→fuerza |

#### 3. Criterio de corrección: tolerancias

```
Error relativo aceleración (GPU f64 vs CPU rayon):  < 1e-10
Error relativo aceleración (GPU f32 vs CPU):         < 1e-5
Error relativo P(k) PM-GPU vs PM-CPU:               < 1e-8 (roundtrip)
```

---

### Suite de validación (V1)

Todos los tests deben agregarse en `crates/gadget-ng-gpu/tests/`.
Se pueden correr con `cargo test -p gadget-ng-gpu --features cuda`.
Los tests sin GPU real deben marcarse `#[ignore]` y documentar
cómo correrlos con `cargo test -- --ignored`.

#### V1-T1: Benchmark GPU vs CPU (cuantitativo)

```rust
// tests/v1_bench_gpu_cpu.rs
//! Verifica que el kernel CUDA reproduce la aceleración directa CPU
//! con error relativo < 1e-10 para N=1024 partículas Plummer.
//!
//! Correr: cargo test -p gadget-ng-gpu --features cuda -- --ignored v1_bench_gpu_cpu

#[test]
#[ignore = "requiere GPU física"]
fn gpu_matches_cpu_direct_gravity_n1024() {
    let n = 1024;
    let eps = 0.01;
    let (pos, mass) = random_plummer_sample(n, seed=42);

    let accel_cpu = direct_gravity_cpu(&pos, &mass, eps);
    let accel_gpu = CudaDirectGravity::new().compute(&pos, &mass, eps);

    for i in 0..n {
        let rel_err = (accel_cpu[i] - accel_gpu[i]).norm() / accel_cpu[i].norm().max(1e-30);
        assert!(rel_err < 1e-10, "partícula {i}: rel_err = {rel_err:.3e}");
    }
}
```

#### V1-T2: Escalado de rendimiento (weak scaling)

```rust
/// Verifica que la GPU es >= 5× más rápida que CPU serial para N >= 4096.
/// El número exacto depende del hardware; documentar en docs/reports/.
#[test]
#[ignore = "requiere GPU física"]
fn gpu_speedup_over_cpu_serial_weak_scaling() {
    for &n in &[1024_usize, 4096, 16384] {
        let (pos, mass) = random_plummer_sample(n, seed=0);
        let t_cpu = time(|| direct_gravity_cpu(&pos, &mass, 0.01));
        let t_gpu = time(|| CudaDirectGravity::new().compute(&pos, &mass, 0.01));
        let speedup = t_cpu / t_gpu;
        println!("N={n}: CPU={t_cpu:.3}s GPU={t_gpu:.3}s speedup={speedup:.1}x");
        if n >= 4096 { assert!(speedup > 5.0, "speedup insuficiente: {speedup:.1}x"); }
    }
}
```

#### V1-T3: PM-GPU roundtrip (si `pm_gpu` feature activo)

```rust
/// Aplica FFT 3D forward + backward y verifica que se recupera la señal
/// con error máximo < 1e-8 (ruido numérico FFT de doble precisión).
#[test]
#[ignore = "requiere GPU con CUFFT"]
#[cfg(feature = "pm_gpu")]
fn pm_gpu_roundtrip_fft() {
    let nm = 64_usize;
    let signal: Vec<f64> = (0..nm.pow(3)).map(|i| (i as f64 * 0.01).sin()).collect();
    let roundtrip = PmGpuSolver::fft_roundtrip(&signal, nm);
    let max_err = signal.iter().zip(&roundtrip).map(|(a,b)| (a-b).abs()).fold(0f64, f64::max);
    assert!(max_err < 1e-8, "error FFT roundtrip: {max_err:.3e}");
}
```

#### V1-T4: P(k) PM-GPU vs PM-CPU

```rust
/// Calcula el espectro de potencias de un campo de densidad con PM-GPU
/// y lo compara con la referencia CPU. Error relativo en cada bin < 1%.
#[test]
#[ignore = "requiere GPU con CUFFT"]
#[cfg(feature = "pm_gpu")]
fn power_spectrum_pm_gpu_matches_pm_cpu() {
    let nm = 128;
    let (particles, box_size) = synthetic_pk_field(nm, seed=7);
    let pk_cpu = pm_power_spectrum_cpu(&particles, nm, box_size);
    let pk_gpu = PmGpuSolver::new(nm).power_spectrum(&particles, box_size);
    for (k_bin, (cpu, gpu)) in pk_cpu.iter().zip(&pk_gpu).enumerate() {
        let rel = (cpu - gpu).abs() / cpu.abs().max(1e-40);
        assert!(rel < 0.01, "bin {k_bin}: rel_err = {rel:.3e}");
    }
}
```

#### V1-T5: Conservación de energía con timestep GPU

```rust
/// Integra N=256 partículas por 100 pasos con solver GPU y verifica
/// que la energía total se conserva con drift < 0.1%.
#[test]
#[ignore = "requiere GPU física"]
fn energy_conservation_gpu_integrator_n256_100steps() {
    let n = 256;
    let dt = 1e-3;
    let mut sim = GpuSimulation::new(n, seed=99);
    let e0 = sim.total_energy();
    for _ in 0..100 { sim.step(dt); }
    let e1 = sim.total_energy();
    let drift = ((e1 - e0) / e0.abs()).abs();
    assert!(drift < 1e-3, "drift de energía: {drift:.4e}");
}
```

#### V1-T6: Consistencia entre feature `cuda` y `wgpu` (sin GPU)

```rust
/// Ejecutable en CI sin GPU. Verifica que ambos backends producen el mismo
/// resultado en una simulación tiny (N=16) usando el solver CPU como referencia.
#[test]
fn both_backends_agree_with_cpu_n16() {
    let n = 16;
    let (pos, mass) = random_plummer_sample(n, seed=1);
    let accel_cpu   = direct_gravity_cpu(&pos, &mass, 0.1);
    let accel_wgpu  = WgpuDirectGravity::new().compute(&pos, &mass, 0.1);
    // CUDA solo si disponible; de lo contrario skip.
    let accel_cuda_opt = CudaDirectGravity::try_new().map(|g| g.compute(&pos, &mass, 0.1));
    for i in 0..n {
        let err_wgpu = (accel_cpu[i] - accel_wgpu[i]).norm();
        assert!(err_wgpu < 1e-4, "wgpu[{i}]: {err_wgpu:.3e}");
        if let Some(ref cuda) = accel_cuda_opt {
            let err_cuda = (accel_cpu[i] - cuda[i]).norm();
            assert!(err_cuda < 1e-10, "cuda[{i}]: {err_cuda:.3e}");
        }
    }
}
```

### Comando de validación completa V1

```bash
# Sin GPU (CI):
cargo test -p gadget-ng-gpu

# Con GPU NVIDIA (workstation/cluster):
cargo test -p gadget-ng-gpu --features cuda -- --include-ignored

# Con PM-GPU:
cargo test -p gadget-ng-gpu --features cuda,pm_gpu -- --include-ignored

# Benchmark Criterion (genera HTML en target/criterion/):
cargo bench -p gadget-ng-gpu --features cuda --bench gpu_vs_cpu
```

---

<a name="v2"></a>
## V2 — Block timesteps acoplados a árbol distribuido MPI

### Estado actual

El engine tiene dos paths **mutuamente excluyentes** actualmente:

| Path | Flag activo | Limitación |
|------|------------|-----------|
| `use_hierarchical_let` | `hierarchical && !cosmo && MPI > 1` | Sin cosmología |
| `use_sfc_let_cosmo` | `cosmo && !hierarchical && MPI > 1` | Sin block timesteps |
| Newtoniano serial | `hierarchical && MPI = 1` | Serial |

El objetivo es habilitar simultáneamente `hierarchical + cosmo + MPI > 1`.

Referencia clave en `engine.rs`:
```rust
// línea 1064
let use_hierarchical_let = cfg.gravity.solver == SolverKind::BarnesHut
    && cfg.timestep.hierarchical
    && !cfg.cosmology.enabled   // ← barrera a eliminar
    && rt.size() > 1
    && !cfg.performance.force_allgather_fallback;
```

### Diseño de implementación

#### Paso 1: Nuevo flag en engine.rs

```rust
// Sustituye use_hierarchical_let
let use_hierarchical_let_cosmo = cfg.gravity.solver == SolverKind::BarnesHut
    && cfg.timestep.hierarchical
    && cfg.cosmology.enabled
    && rt.size() > 1
    && !cfg.performance.force_allgather_fallback;

// Path newtoniano jerárquico (sin cosmo, como hoy)
let use_hierarchical_let_newton = cfg.gravity.solver == SolverKind::BarnesHut
    && cfg.timestep.hierarchical
    && !cfg.cosmology.enabled
    && rt.size() > 1
    && !cfg.performance.force_allgather_fallback;
```

#### Paso 2: Actualizar hierarchical_kdk_step en gadget-ng-integrators

El método `hierarchical_kdk_step` ya acepta `cosmo: Option<(&CosmologyParams, &mut f64)>`.
Hay que verificar que el intercambio de halos SFC dentro del closure de fuerzas
use los factores drift/kick cosmológicos (`CosmoFactors`) por nivel.

```rust
// gadget-ng-integrators/src/hierarchical.rs
pub fn hierarchical_kdk_step(
    particles: &mut [Particle],
    dt_base: f64,
    n_levels: usize,
    force_fn: &mut dyn FnMut(&mut [Particle], Option<&CosmoDiag>),
    cosmo: Option<(&CosmologyParams, &mut f64)>,  // ya existe
    //  ↑ pasar a_current para drift/kick por sub-nivel
) -> HierarchicalStepMeta { ... }
```

Cambio requerido: el closure `force_fn` dentro del loop de niveles debe invocar
`exchange_halos_sfc` (árbol distribuido) antes de evaluar fuerzas locales.

#### Paso 3: Snapshot de HierarchicalState con cosmo

`HierarchicalState` ya es `Serialize/Deserialize`. Solo hay que guardar también
`a_current` en el meta del snapshot para restaurar correctamente al hacer
resume de una corrida cosmológica jerárquica.

### Suite de validación (V2)

Agregar en `crates/gadget-ng-physics/tests/v2_hierarchical_cosmo.rs`.

#### V2-T1: Invariante de masa — 2 ranks MPI, 10 pasos, cosmo+jerárquico

```rust
/// Con MPI=2, cosmología ΛCDM y block timesteps (n_levels=4),
/// la masa total local + remota debe conservarse exactamente.
/// Correr: mpirun -n 2 cargo test --test v2_hierarchical_cosmo -- v2_mass_conserved
#[test]
fn v2_mass_conserved_mpi2_hierarchical_cosmo_10steps() {
    let cfg = RunConfig {
        cosmology: CosmologySection { enabled: true, omega_m: 0.3, omega_lambda: 0.7,
                                      h0: 0.7, a_init: 0.1, periodic: false, ..Default::default() },
        timestep: TimestepSection { hierarchical: true, eta: 0.025, max_level: 4, ..Default::default() },
        gravity: GravitySection { solver: SolverKind::BarnesHut, theta: 0.5, ..Default::default() },
        performance: PerformanceSection { force_allgather_fallback: false, ..Default::default() },
        ..Default::default()
    };
    let particles = plummer_sphere(512, seed=0);
    let (mass0, energy0) = run_and_measure(&cfg, &particles, n_steps=10);
    let (mass1, _)       = run_and_measure(&cfg, &particles, n_steps=10);
    assert_eq!(mass0, mass1, "masa no conservada");
}
```

#### V2-T2: Conservación de energía total — 50 pasos cosmológicos jerárquicos

```rust
/// Energía total (cinética + potencial + término cosmológico a²)
/// no debe derivar > 0.5% en 50 pasos con leapfrog + block timesteps + cosmo.
#[test]
fn v2_energy_drift_cosmo_hierarchical_50steps() {
    let n_steps = 50;
    let (e0, e1) = run_cosmo_hierarchical_energy_check(n_steps, mpi_ranks=1);
    let drift = ((e1 - e0) / e0.abs()).abs();
    assert!(drift < 5e-3, "drift de energía cosmo+jerárquico: {drift:.4e}");
}
```

#### V2-T3: Reproducibilidad serial vs MPI=2

```rust
/// La posición final de la partícula con id=0 debe coincidir entre
/// una corrida serial (MPI=1) y una distribuida (MPI=2) hasta 1e-10.
/// Prueba que el exchange_halos_sfc no rompe el determinismo.
#[test]
fn v2_reproducible_serial_vs_mpi2_10steps() {
    let pos_serial = run_and_get_position(mpi_ranks=1, n_steps=10, particle_id=0);
    let pos_mpi2   = run_and_get_position(mpi_ranks=2, n_steps=10, particle_id=0);
    let err = (pos_serial - pos_mpi2).norm();
    assert!(err < 1e-10, "posición no reproducible serial vs MPI=2: {err:.3e}");
}
```

#### V2-T4: Escalado fuerte (strong scaling) — 1 → 4 → 8 ranks

```rust
/// El tiempo de wall-clock para N=4096, 20 pasos debe reducirse al menos
/// un 40% al pasar de 1 a 4 ranks (eficiencia paralela >= 40%).
/// Documentar resultado real en docs/reports/.
#[test]
#[ignore = "requiere >= 4 cores MPI"]
fn v2_strong_scaling_4096_20steps() {
    let t1 = wall_time_run(mpi_ranks=1, n=4096, steps=20);
    let t4 = wall_time_run(mpi_ranks=4, n=4096, steps=20);
    let efficiency = t1 / (4.0 * t4);
    println!("Strong scaling efficiency 1→4: {:.1}%", efficiency * 100.0);
    assert!(efficiency > 0.40, "eficiencia paralela insuficiente: {:.1}%", efficiency*100.0);
}
```

#### V2-T5: Correctitud del level-sync cosmológico

```rust
/// En corrida cosmológica jerárquica, el factor de escala a(t) al final
/// debe coincidir con la integración serial de Friedmann directa (RK4)
/// con error relativo < 1e-6.
#[test]
fn v2_scale_factor_agreement_hierarchical_vs_friedmann() {
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.7);
    let a_init = 0.1;
    let n_steps = 100;
    let dt = 1e-4;

    let a_friedmann = integrate_friedmann_rk4(&cosmo, a_init, n_steps as f64 * dt);
    let a_hier      = run_hierarchical_cosmo_and_get_a(a_init, n_steps, dt);

    let rel_err = ((a_hier - a_friedmann) / a_friedmann).abs();
    assert!(rel_err < 1e-6, "factor de escala: rel_err = {rel_err:.3e}");
}
```

#### V2-T6: Resume checkpoint con cosmo+jerárquico

```rust
/// Guarda checkpoint en el paso 25, reanuda, continúa hasta 50.
/// Posición final debe ser idéntica a la corrida ininterrumpida de 50 pasos.
#[test]
fn v2_checkpoint_resume_cosmo_hierarchical() {
    let pos_full      = run_cosmo_hier_full(n_steps=50);
    let pos_resumed   = run_cosmo_hier_with_resume(stop_at=25, resume_to=50);
    let err = (pos_full - pos_resumed).norm();
    assert!(err < 1e-12, "divergencia tras resume: {err:.3e}");
}
```

### Comando de validación completa V2

```bash
# Tests unitarios (sin MPI):
cargo test -p gadget-ng-physics --test v2_hierarchical_cosmo

# Tests MPI (requiere mpirun):
mpirun -n 2 cargo test --test v2_hierarchical_cosmo -- v2_reproducible_serial_vs_mpi2
mpirun -n 4 cargo test --test v2_hierarchical_cosmo -- --ignored v2_strong_scaling

# Corrida de referencia larga (benchmark):
cargo run --release -- --config docs/runbooks/cosmo_hierarchical_mpi.toml
```

---

<a name="v3"></a>
## V3 — ICs MHD cosmológicas + validaciones cuantitativas

### Estado actual

Las ICs cosmológicas Zel'dovich/2LPT existen en `ic_zeldovich.rs`.
Los módulos MHD (campos B, RMHD, turbulencia, reconexión, Braginskii) están
implementados y probados con tests de correctitud cualitativa (Phases 133-149).

Pendiente: generación de ICs con campo magnético primordial y validación
**cuantitativa** contra soluciones analíticas.

### Diseño de implementación

#### 3.1 ICs MHD primordiales

```rust
// crates/gadget-ng-core/src/ic_mhd.rs (nuevo)

/// Genera campo magnético inicial con espectro de potencias B(k) ∝ k^n_B.
/// Modo más simple: campo uniforme B0 en dirección z (tangled field = 0).
pub fn primordial_bfield_ic(
    particles: &mut [Particle],
    b0: f64,           // |B| medio en unidades del código
    spectral_index: f64, // n_B (típico: -2.9 para nearly scale-invariant)
    seed: u64,
) { ... }

/// Normaliza |B| para que la presión magnética no domine sobre la térmica:
/// beta_plasma = P_gas / P_mag >> 1 en las ICs (e.g. beta > 100).
pub fn check_plasma_beta(particles: &[Particle]) -> f64 { ... }
```

#### 3.2 Onda de Alfvén cortante (shear Alfvén wave)

Solución analítica:
```
B_y(x, t) = B0 * sin(k·x - ω·t)
ω = k · v_A   donde v_A = B0 / sqrt(4π·ρ)
```

#### 3.3 Onda magnetosónica

Solución analítica:
```
v_x(x, t) = δv · sin(k·x - ω_ms·t)
ρ(x, t)   = ρ0 + δρ · sin(k·x - ω_ms·t)
ω_ms = k · sqrt(v_A² + c_s²)
```

---

### Suite de validación (V3)

Agregar en `crates/gadget-ng-physics/tests/v3_mhd_validation.rs`.

#### V3-T1: Velocidad de Alfvén — convergencia con resolución

```rust
/// Establece una onda de Alfvén 1D con k = 2π/L y mide la frecuencia
/// numérica ω_num comparándola con ω_ana = k·v_A.
/// El error relativo debe converger como O(h²) con la resolución.
///
/// Referencia: Dedner et al. 2002 (divergence cleaning); Stone et al. 2008.
#[test]
fn v3_alfven_wave_frequency_converges_quadratically() {
    let b0 = 1.0_f64;
    let rho0 = 1.0_f64;
    let v_alfven = b0 / rho0.sqrt();      // unidades del código (4π=1)

    for &n in &[32_usize, 64, 128] {
        let h = 1.0 / n as f64;
        let k = 2.0 * std::f64::consts::PI;
        let omega_ana = k * v_alfven;
        let omega_num = simulate_alfven_wave_frequency(n, b0, rho0, n_periods=2.0);
        let rel_err = ((omega_num - omega_ana) / omega_ana).abs();
        println!("N={n} h={h:.4} ω_num={omega_num:.6} ω_ana={omega_ana:.6} err={rel_err:.3e}");
        // Criterio: error < 1% para N=128
        if n == 128 { assert!(rel_err < 0.01, "onda Alfvén: rel_err={rel_err:.3e}"); }
    }
}
```

#### V3-T2: Amortiguamiento de onda Alfvén con viscosidad Braginskii

```rust
/// Con viscosidad de Braginskii activada, la amplitud de la onda de Alfvén
/// debe decaer exponencialmente como A(t) ∝ exp(-γ·t).
/// γ_ana = 2·η_⊥·k²/(ρ·v_A) (disipación por viscosidad anisotrópica).
#[test]
fn v3_alfven_wave_damping_braginskii() {
    let eta = 0.01_f64;
    let k = 2.0 * std::f64::consts::PI;
    let rho = 1.0_f64;
    let v_a = 1.0_f64;
    let gamma_ana = 2.0 * eta * k * k / (rho * v_a);

    let (times, amplitudes) = simulate_alfven_with_braginskii(eta, n_periods=4);
    let gamma_num = fit_exponential_decay(&times, &amplitudes);

    let rel_err = ((gamma_num - gamma_ana) / gamma_ana).abs();
    assert!(rel_err < 0.05, "tasa de amortiguamiento Braginskii: {rel_err:.3e}");
}
```

#### V3-T3: Onda magnetosónica — dispersión cuantitativa

```rust
/// La velocidad de fase de una onda magnetosónica lenta y rápida debe
/// coincidir con v_ms = sqrt(v_A² + c_s²) con error < 1%.
#[test]
fn v3_magnetosonic_wave_phase_velocity() {
    let b0: f64  = 1.0;
    let gamma: f64 = 5.0 / 3.0;
    let rho0: f64  = 1.0;
    let p0: f64    = 0.6;   // c_s² = gamma·p/rho
    let c_s   = (gamma * p0 / rho0).sqrt();
    let v_a   = b0 / rho0.sqrt();
    let v_ms  = (v_a * v_a + c_s * c_s).sqrt();

    let v_num = simulate_magnetosonic_phase_velocity(b0, rho0, p0, gamma, n=64);
    let rel_err = ((v_num - v_ms) / v_ms).abs();
    assert!(rel_err < 0.01, "velocidad magnetosónica: {rel_err:.3e}  (ana={v_ms:.4} num={v_num:.4})");
}
```

#### V3-T4: Conservación de flujo magnético (flux-freeze)

```rust
/// En ausencia de resistividad, el flujo Φ = ∫ B·dA debe conservarse.
/// Se mide en una región de sección transversal fija durante 10 periodos
/// de Alfvén. Derivación < 0.1%.
///
/// Referencia directa: test de flux-freeze del Phase 147 extendido a campo cosmo.
#[test]
fn v3_flux_freeze_cosmological_ic() {
    let mut particles = setup_mhd_cosmo_ic(n=256, b0=0.5, a_init=0.1);
    let flux0 = measure_magnetic_flux(&particles, region=UnitCube);
    run_mhd_cosmo(&mut particles, n_steps=100, dt=1e-3);
    let flux1 = measure_magnetic_flux(&particles, region=UnitCube);
    let drift = ((flux1 - flux0) / flux0.abs()).abs();
    assert!(drift < 1e-3, "flux-freeze derivado: {drift:.4e}");
}
```

#### V3-T5: Perfil de β_plasma en ICs cosmológicas con B primordial

```rust
/// Las ICs MHD cosmológicas deben tener β_plasma >> 1 (el campo B
/// no debe dominar sobre la presión térmica en el universo temprano).
/// Típico: β ~ 10⁶ para B0 comoving ~ nGauss.
#[test]
fn v3_plasma_beta_cosmological_ic_large() {
    let particles = generate_cosmo_mhd_ic(
        n=512, a_init=0.02, omega_m=0.3, omega_lambda=0.7, b0_nGauss=1.0,
    );
    let beta = check_plasma_beta(&particles);
    assert!(beta > 1e4, "β_plasma demasiado bajo en ICs: β={beta:.2e}");
}
```

#### V3-T6: Comparación P(k) con ΛCDM+MHD vs puro ΛCDM

```rust
/// El espectro de potencias de un box con campo B primordial debe
/// coincidir con ΛCDM puro en escalas grandes (k < 0.5 h/Mpc) y
/// mostrar supresión en escalas pequeñas (resistividad de disipación).
///
/// Criterio cuantitativo: |P_MHD(k)/P_ΛCDM(k) - 1| < 1% para k < 0.5.
#[test]
fn v3_pk_mhd_agrees_with_lcdm_large_scales() {
    let pk_lcdm = run_cosmo_no_mhd_and_get_pk(n=256, z_final=0.0);
    let pk_mhd  = run_cosmo_with_mhd_ic_and_get_pk(n=256, z_final=0.0, b0=1e-9);
    for (k, (p_ref, p_mhd)) in pk_lcdm.iter().zip(&pk_mhd).enumerate() {
        if k_value(k) < 0.5 {
            let ratio = (p_mhd / p_ref - 1.0).abs();
            assert!(ratio < 0.01, "P(k) MHD vs ΛCDM en k={:.2}: {ratio:.3e}", k_value(k));
        }
    }
}
```

### Comando de validación completa V3

```bash
# Tests analíticos (no requieren MPI ni GPU):
cargo test -p gadget-ng-physics --test v3_mhd_validation

# Con más salida numérica:
cargo test -p gadget-ng-physics --test v3_mhd_validation -- --nocapture

# Corrida de referencia MHD cosmológica:
cargo run --release -- --config docs/runbooks/mhd_cosmo_reference.toml
```

---

<a name="checklist"></a>
## Checklist de ejecución autónoma

Este checklist permite dejar corriendo las validaciones sin supervisión.
Cada línea es un bloque independiente que puede ejecutarse en paralelo.

### Preparación (hacer una vez)

```bash
# Verificar que compile sin errores antes de empezar
cargo check --workspace 2>&1 | tee /tmp/check.log
grep -c "^error" /tmp/check.log && echo "HAY ERRORES" || echo "OK"

# Correr tests existentes para tener baseline
cargo test --workspace --release 2>&1 | tail -20
```

### Bloque A — GPU (requiere hardware NVIDIA/AMD)

```bash
cargo test -p gadget-ng-gpu --features cuda -- --include-ignored \
    2>&1 | tee logs/v1_gpu_tests.log
echo "EXIT: $?" >> logs/v1_gpu_tests.log
```

**Criterio de éxito:** `test result: ok` en todas las líneas de resumen.  
**Si falla:** Ver `logs/v1_gpu_tests.log`, buscar `FAILED` y revisar tolerancias.

### Bloque B — Block timesteps + MPI

```bash
# Primero implementar (ver sección V2 diseño), luego:
cargo test -p gadget-ng-physics --test v2_hierarchical_cosmo \
    2>&1 | tee logs/v2_hier_tests.log

mpirun -n 2 cargo test --test v2_hierarchical_cosmo \
    -- --nocapture 2>&1 | tee logs/v2_mpi2_tests.log

mpirun -n 4 cargo test --test v2_hierarchical_cosmo \
    -- --ignored --nocapture 2>&1 | tee logs/v2_mpi4_tests.log
```

**Criterio de éxito:** Sin `FAILED`. Buscar líneas `drift < 5e-3` y `efficiency > 40%`.

### Bloque C — MHD validaciones cuantitativas

```bash
cargo test -p gadget-ng-physics --test v3_mhd_validation -- --nocapture \
    2>&1 | tee logs/v3_mhd_tests.log
echo "EXIT: $?" >> logs/v3_mhd_tests.log
```

**Criterio de éxito:** Todos los `rel_err` impresos deben estar por debajo de
los umbrales definidos en cada test. Ver tabla resumen al final del log.

### Script de validación completa (poner en background)

```bash
#!/usr/bin/env bash
# validate_all.sh — ejecuta las tres suites y genera reporte

set -e
mkdir -p logs

echo "=== 2026-04-23 === INICIO VALIDACIÓN HPC" | tee logs/summary.log

echo "--- V3: MHD analítico ---" | tee -a logs/summary.log
cargo test -p gadget-ng-physics --test v3_mhd_validation -- --nocapture \
    2>&1 | tee logs/v3.log
grep -E "^test.*ok|^test.*FAILED|rel_err|drift|speedup" logs/v3.log \
    | tee -a logs/summary.log

echo "--- V2: block timesteps ---" | tee -a logs/summary.log
cargo test -p gadget-ng-physics --test v2_hierarchical_cosmo -- --nocapture \
    2>&1 | tee logs/v2.log
grep -E "^test.*ok|^test.*FAILED|drift|efficiency" logs/v2.log \
    | tee -a logs/summary.log

echo "--- V1: GPU (si disponible) ---" | tee -a logs/summary.log
cargo test -p gadget-ng-gpu -- --include-ignored 2>&1 | tee logs/v1.log \
    || echo "GPU no disponible (ok en CI)" | tee -a logs/summary.log
grep -E "^test.*ok|^test.*FAILED|speedup|rel_err" logs/v1.log \
    | tee -a logs/summary.log

echo "=== 2026-04-23 === FIN VALIDACIÓN HPC" | tee -a logs/summary.log
```

Ejecutar en background:
```bash
chmod +x validate_all.sh
nohup ./validate_all.sh &
echo "PID: $!"
# Monitorear:
tail -f logs/summary.log
```

---

## Resumen de tolerancias

| Test | Magnitud validada | Tolerancia |
|------|-------------------|-----------|
| V1-T1 | Error aceleración GPU f64 vs CPU | < 1×10⁻¹⁰ |
| V1-T2 | Speedup GPU vs CPU serial (N≥4096) | > 5× |
| V1-T3 | Error FFT roundtrip PM-GPU | < 1×10⁻⁸ |
| V1-T4 | P(k) PM-GPU vs PM-CPU por bin | < 1% |
| V1-T5 | Drift energía GPU 100 pasos | < 0.1% |
| V2-T2 | Drift energía cosmo+jerárquico 50 pasos | < 0.5% |
| V2-T3 | Reproducibilidad serial vs MPI=2 | < 1×10⁻¹⁰ |
| V2-T4 | Eficiencia strong scaling 1→4 ranks | > 40% |
| V2-T5 | Error factor de escala a(t) vs Friedmann | < 1×10⁻⁶ |
| V2-T6 | Divergencia tras checkpoint+resume | < 1×10⁻¹² |
| V3-T1 | ω Alfvén vs analítico (N=128) | < 1% |
| V3-T2 | Tasa de amortiguamiento Braginskii | < 5% |
| V3-T3 | Velocidad magnetosónica vs analítico | < 1% |
| V3-T4 | Flux-freeze cosmológico 10 periodos | < 0.1% |
| V3-T5 | β_plasma en ICs MHD cosmológicas | > 10⁴ |
| V3-T6 | P(k) MHD vs ΛCDM en k<0.5 h/Mpc | < 1% |

---

## Orden de implementación recomendado

**Si el objetivo es máximo impacto físico** (sin hardware GPU especial):

```
V3 (MHD validaciones) → V2 (block+MPI cosmo) → V1 (GPU)
```

**Si hay GPU disponible ahora:**

```
V1 GPU (bloqueante, requiere hardware) en paralelo con V3 (analítico, cualquier máquina)
Luego V2
```

**Estimación de esfuerzo:**
- V3: 2–3 sesiones (solo física y tests, sin HW especial).
- V2: 3–4 sesiones (refactoring engine.rs y jerarquía MPI).
- V1: 4–6 sesiones (toolchain CUDA/HIP es lo más variable).

---

*Documento generado: 2026-04-23. Phases 1–160 completadas.*
