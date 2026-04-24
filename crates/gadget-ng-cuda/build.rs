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
    println!("cargo:rerun-if-changed=cuda/direct_gravity.cu");
    println!("cargo:rerun-if-changed=cuda/direct_gravity.h");
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

    // ── 4. Compilar kernels CUDA ──────────────────────────────────────────────
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR no definida"));
    // CUDA_ARCH puede forzarse manualmente; si no, se auto-detecta con nvidia-smi.
    // Default conservador: sm_60 (Pascal, GTX 10xx) si no hay GPU o no se detecta.
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| detect_cuda_arch());

    let lib_path = out_dir.join("libpm_cuda.a");

    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    // Fuentes a compilar: pm_gravity.cu y direct_gravity.cu
    let kernel_sources = [
        ("pm_gravity",     "cuda/pm_gravity.cu"),
        ("direct_gravity", "cuda/direct_gravity.cu"),
    ];

    let mut obj_paths: Vec<PathBuf> = Vec::new();

    for (name, src_rel) in &kernel_sources {
        let src = manifest_dir.join(src_rel);
        let obj = out_dir.join(format!("{name}.o"));

        // nvcc -O3 -arch=<sm> -Xcompiler -fPIC -c <kernel.cu> -o <kernel.o>
        let status = Command::new(&nvcc)
            .args([
                "-O3",
                &format!("-arch={arch}"),
                "-Xcompiler",
                "-fPIC",
                "-std=c++14",
                "-c",
                src.to_str().expect("kernel path"),
                "-o",
                obj.to_str().expect("obj path"),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                println!("cargo:warning=nvcc falló al compilar {name} (código {s}). CUDA deshabilitado.");
                println!("cargo:rustc-cfg=cuda_unavailable");
                return;
            }
            Err(e) => {
                println!("cargo:warning=No se pudo ejecutar nvcc para {name}: {e}. CUDA deshabilitado.");
                println!("cargo:rustc-cfg=cuda_unavailable");
                return;
            }
        }
        obj_paths.push(obj);
    }

    // ar rcs libpm_cuda.a pm_gravity.o direct_gravity.o
    let mut ar_args = vec!["rcs".to_string(), lib_path.to_str().expect("lib path").to_string()];
    for obj in &obj_paths {
        ar_args.push(obj.to_str().expect("obj path").to_string());
    }
    let status = Command::new("ar").args(&ar_args).status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=ar falló (código {s}). CUDA deshabilitado.");
            println!("cargo:rustc-cfg=cuda_unavailable");
            return;
        }
        Err(e) => {
            println!("cargo:warning=No se pudo ejecutar ar: {e}. CUDA deshabilitado.");
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

    // Los kernels CUDA usan operadores C++ (new, nothrow) — necesitan libstdc++.
    println!("cargo:rustc-link-lib=stdc++");

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

/// Auto-detecta la arquitectura CUDA de la GPU instalada via `nvidia-smi`.
///
/// Convierte compute capability (e.g. "6.1") → flag nvcc (e.g. "sm_61").
/// Si no hay GPU o falla el comando, devuelve `sm_60` como mínimo compatible
/// con Pascal (GTX 10xx, Quadro P series).
fn detect_cuda_arch() -> String {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();

    if let Ok(o) = out {
        if o.status.success() {
            let cap = String::from_utf8_lossy(&o.stdout);
            // Tomar la primera GPU; eliminar el punto (6.1 → sm_61)
            if let Some(line) = cap.lines().next() {
                let sm = line.trim().replace('.', "");
                if !sm.is_empty() {
                    println!(
                        "cargo:warning=GPU detectada: compute capability {}, compilando para sm_{sm}",
                        line.trim()
                    );
                    return format!("sm_{sm}");
                }
            }
        }
    }

    println!("cargo:warning=No se pudo detectar GPU; usando sm_60 (Pascal) por defecto.");
    "sm_60".to_string()
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
