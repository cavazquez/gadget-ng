//! Build script para gadget-ng-hip.
//!
//! # Flujo de detección
//!
//! 1. `HIP_SKIP=1`          → skip silencioso (CI sin GPU): emite `hip_unavailable`.
//! 2. `hipcc` no encontrado  → emite warning + `hip_unavailable`.
//! 3. rocFFT no encontrada   → emite warning + `hip_unavailable`.
//! 4. Todo OK                → compila `hip/pm_gravity.hip` con hipcc, crea librería
//!                            estática `libpm_hip.a` y enlaza con rocFFT + hipruntime.
//!
//! # Variables de entorno respetadas
//!
//! - `HIP_SKIP`   — si `1`, salta todo sin error.
//! - `ROCM_PATH`  — ruta raíz de ROCm (e.g. `/opt/rocm`).
//!                  Si no está definida, se busca hipcc en PATH.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=hip/pm_gravity.hip");
    println!("cargo:rerun-if-changed=hip/pm_gravity.h");
    println!("cargo:rerun-if-env-changed=HIP_SKIP");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    // Declarar el cfg personalizado para silenciar el lint `unexpected_cfgs`.
    println!("cargo::rustc-check-cfg=cfg(hip_unavailable)");

    // ── 1. Skip si HIP_SKIP=1 ─────────────────────────────────────────────────
    if std::env::var("HIP_SKIP").as_deref() == Ok("1") {
        println!("cargo:warning=HIP_SKIP=1: HipPmSolver deshabilitado (stubs activos)");
        println!("cargo:rustc-cfg=hip_unavailable");
        return;
    }

    // ── 2. Detectar hipcc ─────────────────────────────────────────────────────
    let rocm_path = std::env::var("ROCM_PATH").ok().map(PathBuf::from);
    let hipcc = find_hipcc(rocm_path.as_deref());

    let Some(hipcc) = hipcc else {
        println!(
            "cargo:warning=hipcc no encontrado (instalar ROCm o definir ROCM_PATH). \
             HipPmSolver deshabilitado."
        );
        println!("cargo:rustc-cfg=hip_unavailable");
        return;
    };

    // ── 3. Detectar rocFFT ────────────────────────────────────────────────────
    let rocfft_lib_dir = find_rocfft_lib(rocm_path.as_deref());

    let Some(rocfft_lib_dir) = rocfft_lib_dir else {
        println!(
            "cargo:warning=rocFFT no encontrada (librocfft.so). \
             HipPmSolver deshabilitado."
        );
        println!("cargo:rustc-cfg=hip_unavailable");
        return;
    };

    // ── 4. Compilar kernel HIP ────────────────────────────────────────────────
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR no definida"));
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    let kernel_src = manifest_dir.join("hip/pm_gravity.hip");
    let obj_path = out_dir.join("pm_gravity_hip.o");
    let lib_path = out_dir.join("libpm_hip.a");

    // hipcc -O3 -fPIC -c <kernel.hip> -o <kernel.o>
    let status = Command::new(&hipcc)
        .args([
            "-O3",
            "-fPIC",
            "-std=c++14",
            "-c",
            kernel_src.to_str().expect("kernel path"),
            "-o",
            obj_path.to_str().expect("obj path"),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=hipcc falló (código {s}). HipPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=hip_unavailable");
            return;
        }
        Err(e) => {
            println!("cargo:warning=No se pudo ejecutar hipcc: {e}. HipPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=hip_unavailable");
            return;
        }
    }

    // ar rcs libpm_hip.a pm_gravity_hip.o
    let status = Command::new("ar")
        .args([
            "rcs",
            lib_path.to_str().expect("lib path"),
            obj_path.to_str().expect("obj path"),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=ar falló (código {s}). HipPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=hip_unavailable");
            return;
        }
        Err(e) => {
            println!("cargo:warning=No se pudo ejecutar ar: {e}. HipPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=hip_unavailable");
            return;
        }
    }

    // ── 5. Directivas de enlazado ─────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pm_hip");

    println!(
        "cargo:rustc-link-search=native={}",
        rocfft_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=rocfft");
    println!("cargo:rustc-link-lib=hipruntime");

    // Directorio adicional de hip runtime si ROCm está en ruta estándar.
    if let Some(ref rp) = rocm_path {
        let hip_lib = rp.join("lib");
        if hip_lib.exists() {
            println!("cargo:rustc-link-search=native={}", hip_lib.display());
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Busca `hipcc` primero en `$ROCM_PATH/bin`, luego en PATH y rutas estándar.
fn find_hipcc(rocm_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(rp) = rocm_path {
        let candidate = rp.join("bin/hipcc");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Ok(out) = Command::new("which").arg("hipcc").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return Some(PathBuf::from(s));
            }
        }
    }

    for candidate in ["/opt/rocm/bin/hipcc", "/usr/bin/hipcc"] {
        if Path::new(candidate).exists() {
            return Some(PathBuf::from(candidate));
        }
    }

    None
}

/// Busca el directorio que contiene `librocfft.so`.
fn find_rocfft_lib(rocm_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(rp) = rocm_path {
        let candidate = rp.join("lib");
        if candidate.join("librocfft.so").exists() {
            return Some(candidate);
        }
    }

    for dir in ["/opt/rocm/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"] {
        let p = Path::new(dir);
        if p.join("librocfft.so").exists() {
            return Some(p.to_path_buf());
        }
    }

    if let Ok(out) = Command::new("pkg-config")
        .args(["--libs-only-L", "rocfft"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let dir = s.trim_start_matches("-L");
            if !dir.is_empty() {
                return Some(PathBuf::from(dir));
            }
        }
    }

    None
}
