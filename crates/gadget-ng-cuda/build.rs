//! Build script para gadget-ng-cuda.
//!
//! # Flujo de detección
//!
//! 1. `CUDA_SKIP=1`          → skip silencioso (CI sin GPU): emite `cuda_unavailable`.
//! 2. `nvcc` no encontrado   → emite warning + `cuda_unavailable`.
//! 3. cuFFT no encontrada    → emite warning + `cuda_unavailable`.
//! 4. Todo OK → compila `cuda/pm_gravity.cu` con nvcc, crea librería
//!    estática `libpm_cuda.a` y enlaza con cuFFT + cudart.
//!
//! # Variables de entorno respetadas
//!
//! - `CUDA_SKIP`  — si `1`, salta todo sin error.
//! - `CUDA_PATH`  — ruta raíz del toolkit (e.g. `/usr/local/cuda`).
//!   Si no está definida, se busca nvcc en PATH.
//! - `CUDA_ARCH`  — arquitectura de compilación (default: `sm_80`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/pm_gravity.cu");
    println!("cargo:rerun-if-changed=cuda/pm_gravity.h");
    println!("cargo:rerun-if-env-changed=CUDA_SKIP");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // Declarar el cfg personalizado para silenciar el lint `unexpected_cfgs`.
    println!("cargo::rustc-check-cfg=cfg(cuda_unavailable)");

    // ── 1. Skip si CUDA_SKIP=1 ────────────────────────────────────────────────
    if std::env::var("CUDA_SKIP").as_deref() == Ok("1") {
        println!("cargo:warning=CUDA_SKIP=1: CudaPmSolver deshabilitado (stubs activos)");
        println!("cargo:rustc-cfg=cuda_unavailable");
        return;
    }

    // ── 2. Detectar nvcc ──────────────────────────────────────────────────────
    let cuda_path = std::env::var("CUDA_PATH").ok().map(PathBuf::from);
    let nvcc = find_nvcc(cuda_path.as_deref());

    let Some(nvcc) = nvcc else {
        println!(
            "cargo:warning=nvcc no encontrado (instalar CUDA Toolkit o definir CUDA_PATH). \
             CudaPmSolver deshabilitado."
        );
        println!("cargo:rustc-cfg=cuda_unavailable");
        return;
    };

    // ── 3. Detectar cuFFT ─────────────────────────────────────────────────────
    let cufft_lib_dir = find_cufft_lib(cuda_path.as_deref());

    let Some(cufft_lib_dir) = cufft_lib_dir else {
        println!(
            "cargo:warning=cuFFT no encontrada (libcufft.so). \
             CudaPmSolver deshabilitado."
        );
        println!("cargo:rustc-cfg=cuda_unavailable");
        return;
    };

    // ── 4. Compilar kernel CUDA ───────────────────────────────────────────────
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR no definida"));
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

    let obj_path = out_dir.join("pm_gravity.o");
    let lib_path = out_dir.join("libpm_cuda.a");

    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let kernel_src = manifest_dir.join("cuda/pm_gravity.cu");

    // nvcc -O3 -arch=<sm> -Xcompiler -fPIC -c <kernel.cu> -o <kernel.o>
    let status = Command::new(&nvcc)
        .args([
            "-O3",
            &format!("-arch={arch}"),
            "-Xcompiler",
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
            println!("cargo:warning=nvcc falló (código {s}). CudaPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=cuda_unavailable");
            return;
        }
        Err(e) => {
            println!("cargo:warning=No se pudo ejecutar nvcc: {e}. CudaPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=cuda_unavailable");
            return;
        }
    }

    // ar rcs libpm_cuda.a pm_gravity.o
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
            println!("cargo:warning=ar falló (código {s}). CudaPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=cuda_unavailable");
            return;
        }
        Err(e) => {
            println!("cargo:warning=No se pudo ejecutar ar: {e}. CudaPmSolver deshabilitado.");
            println!("cargo:rustc-cfg=cuda_unavailable");
            return;
        }
    }

    // ── 5. Directivas de enlazado ─────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pm_cuda");

    println!("cargo:rustc-link-search=native={}", cufft_lib_dir.display());
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=cudart");

    // Directorio stubs para cudart en instalaciones headless.
    if let Some(ref cp) = cuda_path {
        let stubs = cp.join("lib64/stubs");
        if stubs.exists() {
            println!("cargo:rustc-link-search=native={}", stubs.display());
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Busca `nvcc` primero en `$CUDA_PATH/bin`, luego en PATH.
fn find_nvcc(cuda_path: Option<&Path>) -> Option<PathBuf> {
    // Opción 1: $CUDA_PATH/bin/nvcc
    if let Some(cp) = cuda_path {
        let candidate = cp.join("bin/nvcc");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Opción 2: nvcc en PATH
    if let Ok(out) = Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return Some(PathBuf::from(s));
            }
        }
    }

    // Opción 3: ruta estándar CUDA
    for candidate in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"] {
        if Path::new(candidate).exists() {
            return Some(PathBuf::from(candidate));
        }
    }

    None
}

/// Busca el directorio que contiene `libcufft.so`.
fn find_cufft_lib(cuda_path: Option<&Path>) -> Option<PathBuf> {
    // Opción 1: $CUDA_PATH/lib64
    if let Some(cp) = cuda_path {
        let candidate = cp.join("lib64");
        if candidate.join("libcufft.so").exists() {
            return Some(candidate);
        }
        // Algunas instalaciones Windows/Linux usan /lib en lugar de /lib64
        let candidate = cp.join("lib");
        if candidate.join("libcufft.so").exists() {
            return Some(candidate);
        }
    }

    // Opción 2: rutas estándar del sistema
    for dir in [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    ] {
        let p = Path::new(dir);
        if p.join("libcufft.so").exists() {
            return Some(p.to_path_buf());
        }
    }

    // Opción 3: pkg-config (último recurso)
    if let Ok(out) = Command::new("pkg-config")
        .args(["--libs-only-L", "cufft"])
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
