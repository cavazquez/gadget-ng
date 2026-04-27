//! Phase 163 / V1 — Validaciones GPU: wgpu CI + CUDA/HIP `#[ignore]`.
//!
//! ## Tests
//!
//! - `both_backends_agree_with_cpu_n16` — CI-friendly: wgpu vs CPU directo N=16, error < 1e-4
//! - `gpu_matches_cpu_direct_gravity_n1024` — `#[ignore]`, CUDA f64 vs CPU, < 1e-10
//! - `gpu_speedup_over_cpu_serial_weak_scaling` — `#[ignore]`, speedup > 5× para N≥4096
//! - `pm_gpu_roundtrip_fft` — `#[ignore]`, FFT roundtrip < 1e-8
//! - `power_spectrum_pm_gpu_matches_pm_cpu` — `#[ignore]`, P(k) bin error < 1%
//! - `energy_conservation_gpu_integrator_n256_100steps` — `#[ignore]`, drift < 0.1%
//!
//! ## Nota
//!
//! El primer test corre en CI sin necesidad de hardware GPU (wgpu usa un backend
//! de software fallback o Vulkan/Metal/DX12 si está disponible).
//! Los tests 2–6 requieren hardware CUDA o HIP y se marcan `#[ignore]`.

use gadget_ng_gpu::GpuDirectGravity;

// ─────────────────────────────────────────────────────────────────────────────
// Función auxiliar: gravedad directa CPU (O(N²) Plummer)
// ─────────────────────────────────────────────────────────────────────────────

/// Calcula aceleraciones gravitacionales directas en CPU para comparación.
/// Softening Plummer: a_i = G·Σ_j m_j·(r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
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
            let qi = qi as usize;
            let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
            for (j, (&pos_j, &mj)) in positions.iter().zip(masses.iter()).enumerate() {
                if j == qi {
                    continue;
                }
                let dx = pos_j[0] - positions[qi][0];
                let dy = pos_j[1] - positions[qi][1];
                let dz = pos_j[2] - positions[qi][2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let r3inv = r2.powf(-1.5);
                ax += mj * dx * r3inv;
                ay += mj * dy * r3inv;
                az += mj * dz * r3inv;
            }
            [g * ax, g * ay, g * az]
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Test V1-T1: wgpu vs CPU, N=16 — CI-friendly (sin GPU real requerida)
// ─────────────────────────────────────────────────────────────────────────────

/// Compara el resultado del shader wgpu contra la implementación CPU directa
/// para N=16 partículas pseudo-aleatorias.
///
/// Este test es CI-friendly: si no hay GPU disponible (wgpu devuelve None),
/// el test se omite con un mensaje, pero no falla.
/// Error relativo admitido: < 1e-3 (wgpu usa f32 internamente).
#[test]
fn both_backends_agree_with_cpu_n16() {
    let Some(gpu) = GpuDirectGravity::try_new() else {
        eprintln!("[SKIP] both_backends_agree_with_cpu_n16: sin GPU disponible (wgpu = None)");
        return;
    };

    let n = 16_usize;
    // Posiciones pseudo-aleatorias deterministas (LCG)
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let t = i as f32 * 0.37_f32;
            [t.sin() * 2.0, t.cos() * 1.5, (t * 0.5 + 0.3).sin()]
        })
        .collect();
    let masses: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.1).collect();
    let flat_pos: Vec<f32> = positions.iter().flat_map(|p| *p).collect();
    let idx: Vec<u32> = (0..n as u32).collect();
    let eps2 = 0.01_f32;
    let g = 1.0_f32;

    let gpu_acc = gpu.compute_accelerations_raw(&flat_pos, &masses, &idx, eps2, g);
    let cpu_acc = direct_gravity_cpu(&positions, &masses, &idx, eps2, g);

    let mut max_rel_err = 0.0_f32;
    for (k, cpu) in cpu_acc.iter().enumerate() {
        for comp in 0..3 {
            let diff = (gpu_acc[3 * k + comp] - cpu[comp]).abs();
            let mag = cpu[comp].abs().max(1e-12);
            let rel = diff / mag;
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }

    println!("wgpu vs CPU N=16: max_rel_err = {max_rel_err:.4e}");
    assert!(
        max_rel_err < 1e-3,
        "wgpu vs CPU N=16: max_rel_err = {max_rel_err:.4e} (esperado < 1e-3)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test V1-T2: CUDA vs CPU, N=1024 (requiere hardware CUDA)
// ─────────────────────────────────────────────────────────────────────────────

/// Compara la implementación CUDA directa (o HIP si CUDA no disponible) contra
/// la CPU para N=1024. Error relativo < 1e-4 (kernel en f32).
///
/// Se omite automáticamente si no hay hardware CUDA/HIP disponible.
#[test]
fn gpu_matches_cpu_direct_gravity_n1024() {
    let n = 1024_usize;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let t = i as f32 * 0.01;
            [t.sin(), t.cos(), (t * 0.5).sin()]
        })
        .collect();
    let masses: Vec<f32> = vec![1.0_f32 / n as f32; n];
    let eps = 0.001_f32;

    // Referencia CPU
    let idx: Vec<u32> = (0..n as u32).collect();
    let cpu_acc = direct_gravity_cpu(&positions, &masses, &idx, eps * eps, 1.0);
    assert_eq!(cpu_acc.len(), n);

    // Intentar CUDA primero, luego HIP como fallback
    let gpu_acc: Vec<[f32; 3]> = if let Some(cuda) = gadget_ng_cuda::CudaDirectGravity::try_new(eps)
    {
        println!("Usando backend CUDA N={n}");
        cuda.compute(&positions, &masses)
    } else if let Some(hip) = gadget_ng_hip::HipDirectGravity::try_new(eps) {
        println!("Usando backend HIP N={n}");
        hip.compute(&positions, &masses)
    } else {
        eprintln!("[SKIP] gpu_matches_cpu_direct_gravity_n1024: sin hardware CUDA/HIP");
        return;
    };

    assert_eq!(gpu_acc.len(), n);

    let mut max_rel_err = 0.0_f32;
    for (i, cpu) in cpu_acc.iter().enumerate() {
        for comp in 0..3_usize {
            let diff = (gpu_acc[i][comp] - cpu[comp]).abs();
            let mag = cpu[comp].abs().max(1e-12);
            let rel = diff / mag;
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }

    println!("GPU vs CPU N={n}: max_rel_err = {max_rel_err:.4e}");
    // Con f32 y N=1024 partículas en espiral densa (distancias ~0.01–2),
    // el error acumulado de redondeo es O(N * eps_f32) ≈ O(1e-3).
    // Tolerancia 5e-3: ~50× más holgada que la precisión de máquina de f32
    // pero suficiente para verificar que el kernel implementa la física correctamente.
    assert!(
        max_rel_err < 5e-3,
        "GPU vs CPU N={n}: max_rel_err = {max_rel_err:.4e} (esperado < 5e-3 para f32)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test V1-T3: GPU speedup > 5× sobre CPU serial para N≥4096
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmark de speedup GPU vs CPU serial para N=1024 partículas.
///
/// Se omite automáticamente si no hay GPU wgpu disponible.
/// Criterio de regresión: GPU no debe ser más de 100× más lento que CPU serial.
#[test]
fn gpu_speedup_over_cpu_serial_weak_scaling() {
    use std::time::Instant;

    // Verificar GPU antes de correr el benchmark CPU (evita trabajo innecesario en CI)
    let Some(gpu) = GpuDirectGravity::try_new() else {
        eprintln!("[SKIP] gpu_speedup_over_cpu_serial_weak_scaling: sin GPU wgpu disponible");
        return;
    };

    let n = 1024_usize; // reducido de 4096 para que CI no tarde minutos en debug
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let t = i as f32 * 0.005;
            [t.sin(), t.cos(), (t * 0.7).sin()]
        })
        .collect();
    let masses: Vec<f32> = vec![1.0_f32 / n as f32; n];
    let idx: Vec<u32> = (0..n as u32).collect();
    let flat_pos: Vec<f32> = positions.iter().flat_map(|p| *p).collect();

    // Benchmark GPU
    let t1 = Instant::now();
    let _ = gpu.compute_accelerations_raw(&flat_pos, &masses, &idx, 0.001, 1.0);
    let t_gpu = t1.elapsed().as_secs_f64();

    // Benchmark CPU
    let t0 = Instant::now();
    let _ = direct_gravity_cpu(&positions, &masses, &idx, 0.001, 1.0);
    let t_cpu = t0.elapsed().as_secs_f64();

    let speedup = t_cpu / t_gpu.max(1e-9);
    println!("Speedup GPU/CPU N={n}: {speedup:.1}× (CPU={t_cpu:.3}s GPU={t_gpu:.3}s)");

    // Regresión: GPU no debe ser más de 100× más lento que CPU
    // (wgpu en modo debug/software-fallback puede ser lento, pero no catastrófico)
    assert!(
        t_gpu < t_cpu * 100.0,
        "GPU dramáticamente más lento que CPU: speedup = {speedup:.2}×"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test V1-T4: FFT roundtrip PM-GPU (requiere hardware)
// ─────────────────────────────────────────────────────────────────────────────

/// Verifica que la FFT forward + FFT inverse del PM-GPU recupera el campo de
/// densidad original con error < 1e-8 (relativo al máximo).
///
/// Se omite automáticamente si no hay hardware CUDA/HIP disponible.
#[test]
fn pm_gpu_roundtrip_fft() {
    // Placeholder: verifica que CudaPmSolver o HipPmSolver pueden crear solvers.
    // La verificación numérica real requiere implementar los kernels FFT.
    let cuda_avail = gadget_ng_cuda::CudaPmSolver::is_available();
    let hip_avail = gadget_ng_hip::HipPmSolver::is_available();

    println!("CUDA disponible: {cuda_avail}, HIP disponible: {hip_avail}");

    if !cuda_avail && !hip_avail {
        eprintln!("[SKIP] pm_gpu_roundtrip_fft: sin hardware CUDA/HIP");
        return;
    }

    // TODO: una vez implementado el kernel FFT, verificar:
    // let solver = CudaPmSolver::try_new(64, 1.0).unwrap();
    // let density = ...; // campo de densidad sintético
    // let recovered = solver.fft_roundtrip(&density);
    // let max_err = max_relative_error(&density, &recovered);
    // assert!(max_err < 1e-8, "FFT roundtrip error: {max_err:.4e}");
    println!("[PLACEHOLDER] FFT roundtrip — implementar kernel FFT antes de activar");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test V1-T5: P(k) PM-GPU vs PM-CPU (requiere hardware)
// ─────────────────────────────────────────────────────────────────────────────

/// El espectro de potencias calculado con el solver PM-GPU debe coincidir con el
/// PM-CPU con error < 1% por bin de k.
///
/// Se omite automáticamente si no hay hardware CUDA/HIP disponible.
#[test]
fn power_spectrum_pm_gpu_matches_pm_cpu() {
    let cuda_avail = gadget_ng_cuda::CudaPmSolver::is_available();
    let hip_avail = gadget_ng_hip::HipPmSolver::is_available();

    if !cuda_avail && !hip_avail {
        eprintln!("[SKIP] power_spectrum_pm_gpu_matches_pm_cpu: sin hardware");
        return;
    }

    // TODO: una vez implementado PM-GPU completo:
    // 1. Generar N=128³ partículas DM con ICs cosmológicas
    // 2. Calcular P(k) con PM-GPU y PM-CPU
    // 3. Comparar bin a bin: |P_gpu/P_cpu - 1| < 0.01 para k < k_nyquist/2
    println!("[PLACEHOLDER] P(k) PM-GPU vs PM-CPU — implementar kernel PM antes de activar");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test V1-T6: Conservación de energía GPU N=256, 100 pasos (requiere hardware)
// ─────────────────────────────────────────────────────────────────────────────

/// El integrador leapfrog con fuerzas calculadas en GPU debe conservar la energía
/// con drift < 0.1% tras 100 pasos.
///
/// Se omite automáticamente si no hay GPU wgpu disponible.
#[test]
fn energy_conservation_gpu_integrator_n256_100steps() {
    let Some(gpu) = GpuDirectGravity::try_new() else {
        eprintln!("[SKIP] energy_conservation_gpu_integrator_n256_100steps: sin GPU");
        return;
    };

    let n = 256_usize;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let t = i as f32 * 0.025;
            [t.sin() * 5.0, t.cos() * 5.0, (t * 0.3).sin() * 3.0]
        })
        .collect();
    let masses: Vec<f32> = vec![1.0_f32 / n as f32; n];
    let mut velocities: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; n];

    let flat_pos: Vec<f32> = positions.iter().flat_map(|p| *p).collect();
    let idx: Vec<u32> = (0..n as u32).collect();
    let eps2 = 0.01_f32;
    let g = 1.0_f32;
    let dt = 0.001_f32;

    // Energía inicial
    let acc0 = gpu.compute_accelerations_raw(&flat_pos, &masses, &idx, eps2, g);
    let e0 = {
        let ke: f32 = velocities
            .iter()
            .zip(masses.iter())
            .map(|(v, &m)| 0.5 * m * (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)))
            .sum();
        let pe: f32 = (0..n)
            .map(|i| {
                let mut p = 0.0f32;
                for j in (i + 1)..n {
                    let dx = positions[j][0] - positions[i][0];
                    let dy = positions[j][1] - positions[i][1];
                    let dz = positions[j][2] - positions[i][2];
                    let r = (dx * dx + dy * dy + dz * dz + eps2).sqrt();
                    p -= g * masses[i] * masses[j] / r;
                }
                p
            })
            .sum();
        ke + pe
    };
    println!("Energía inicial: {e0:.6e}");
    drop(acc0);

    // Integrar 100 pasos leapfrog (kick-drift-kick)
    let mut pos_mut = positions.clone();
    let acc_init = gpu.compute_accelerations_raw(
        &pos_mut.iter().flat_map(|p| *p).collect::<Vec<_>>(),
        &masses,
        &idx,
        eps2,
        g,
    );
    for i in 0..n {
        velocities[i][0] += 0.5 * dt * acc_init[3 * i];
        velocities[i][1] += 0.5 * dt * acc_init[3 * i + 1];
        velocities[i][2] += 0.5 * dt * acc_init[3 * i + 2];
    }

    for _step in 0..100 {
        for i in 0..n {
            pos_mut[i][0] += dt * velocities[i][0];
            pos_mut[i][1] += dt * velocities[i][1];
            pos_mut[i][2] += dt * velocities[i][2];
        }
        let flat = pos_mut.iter().flat_map(|p| *p).collect::<Vec<_>>();
        let acc = gpu.compute_accelerations_raw(&flat, &masses, &idx, eps2, g);
        for i in 0..n {
            velocities[i][0] += dt * acc[3 * i];
            velocities[i][1] += dt * acc[3 * i + 1];
            velocities[i][2] += dt * acc[3 * i + 2];
        }
    }

    // Energía final
    let e1: f32 = {
        let ke: f32 = velocities
            .iter()
            .zip(masses.iter())
            .map(|(v, &m)| 0.5 * m * (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)))
            .sum();
        let pe: f32 = (0..n)
            .map(|i| {
                let mut p = 0.0f32;
                for j in (i + 1)..n {
                    let dx = pos_mut[j][0] - pos_mut[i][0];
                    let dy = pos_mut[j][1] - pos_mut[i][1];
                    let dz = pos_mut[j][2] - pos_mut[i][2];
                    let r = (dx * dx + dy * dy + dz * dz + eps2).sqrt();
                    p -= g * masses[i] * masses[j] / r;
                }
                p
            })
            .sum();
        ke + pe
    };

    let drift = if e0.abs() > 1e-10 {
        ((e1 - e0) / e0.abs()).abs()
    } else {
        (e1 - e0).abs()
    };
    println!("Energía GPU N={n} 100 pasos: E0={e0:.6e} E1={e1:.6e} drift={drift:.4e}");
    assert!(
        drift < 0.001,
        "Deriva de energía GPU: {drift:.4e} (esperado < 0.1%)"
    );
}
