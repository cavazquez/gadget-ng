# Plan de Implementación — 2026-05-07

> Documento vivo con las acciones concretas para cerrar gaps del proyecto `gadget-ng`.
> Prioridad: JOSS → Validación producción → GPU optimización → CI hardening.

---

## 0. Checklist Diario (antes de empezar)

```bash
# 1. Estado limpio
git status

# 2. Tests base pasan
cargo test -p gadget-ng-core
cargo test --workspace --lib
bash scripts/check-physics.sh

# 3. Clippy limpio
cargo clippy --workspace -- -D warnings
cargo clippy --workspace --all-targets -- -D warnings

# 4. Formato
cargo fmt --all -- --check
```

Si alguno falla, **arreglar antes de tocar cualquier otra cosa**.

---

## 1. Paper JOSS (Prioridad: CRÍTICA)

### 1.1 Estado actual
- Draft en `docs/paper/paper.md`
- Faltan: figuras, DOI Zenodo, ORCIDs, compilación PDF, statement of need pulido.

### 1.2 Pasos concretos

#### A. Completar metadatos (`docs/paper/paper.md`)
- [ ] Revisar `title:` y `authors:` — verificar ORCIDs de todos los autores.
- [ ] Agregar `affiliations:` con instituciones actualizadas.
- [ ] En `abstract:` asegurar que mencione: Rust, N-body, SPH, MHD, RT, GPU.
- [ ] En `statement_of_need:` responder explícitamente: *"¿Por qué el campo necesita gadget-ng si ya existe GADGET-4/Arepo/Ramses?"* (respuesta: memoria segura, reproducibilidad bit-a-bit serial, backend GPU unificado wgpu/CUDA/HIP).

#### B. Generar figuras para el paper
Crear script Python en `docs/paper/figures/generate_all.py`:

```python
#!/usr/bin/env python3
"""Genera figuras del paper JOSS. Requiere: pip install matplotlib numpy"""
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

OUT = Path(__file__).parent
DATA = Path("experiments/nbody")

def fig1_validation_phases():
    """Barras de tests pasados por fase V1/V2/V3."""
    ...

def fig2_pk_comparison():
    """P(k) gadget-ng vs CLASS a z=0 para N=128^3."""
    ...

def fig3_gpu_speedup():
    """Speedup GPU vs CPU para distintos N y solvers."""
    ...

def fig4_scaling():
    """Strong/weak scaling MPI."""
    ...

if __name__ == "__main__":
    fig1_validation_phases()
    fig2_pk_comparison()
    fig3_gpu_speedup()
    fig4_scaling()
```

- [ ] `fig1`: contar tests pasados por fase usando `cargo test --workspace -- --format=json` o parsear nombres de archivos en `gadget-ng-physics/tests/`.
- [ ] `fig2`: usar datos de `experiments/nbody/phase38_class_validation/reference/` o regenerar con `cargo test -p gadget-ng-physics phase38 -- --include-ignored --nocapture`.
- [ ] `fig3`: **ver punto 3 GPU**. Necesitamos números reales. Si no hay hardware GPU disponible, dejar placeholder con etiqueta "simulated" y nota al pie.
- [ ] `fig4`: usar datos de `experiments/nbody/phase3_gadget4_benchmark/` o `phase43_dt_treepm_parallel`.

#### C. Compilar PDF
- [ ] Instalar `pandoc` + `latex` si no están: `sudo apt-get install pandoc texlive-latex-base texlive-latex-extra`.
- [ ] Agregar `docs/paper/Makefile`:
  ```makefile
  paper.pdf: paper.md
  	pandoc paper.md -o paper.pdf --template=joss-latex-template
  ```
- [ ] Descargar template LaTeX de JOSS (`https://github.com/openjournals/whedon/raw/master/resources/joss.template`) si no existe.
- [ ] Compilar y verificar que las figuras se incluyen.

#### D. Zenodo + DOI
- [ ] Crear release tag `v0.9.0` (pre-JOSS) o `v1.0.0` si consideramos que es release inicial.
- [ ] Subir a Zenodo (vía GitHub integration) para obtener DOI.
- [ ] Pegar DOI en `paper.md` bajo `doi:`.

#### E. Submit
- [ ] Revisar checklist de JOSS: https://joss.readthedocs.io/en/latest/submitting.html
- [ ] Crear repo `gadget-ng-paper` si es necesario (JOSS acepta paper en repo separado o en `docs/paper/` del repo principal).
- [ ] Abrir issue de submission en https://github.com/openjournals/joss-reviews/issues/new usando el botón "Submit".

---

## 2. Validación de Producción N=128³ (Prioridad: ALTA)

### 2.1 Objetivo
Correr simulación cosmológica completa desde z=99 hasta z=0 con N=128³, y comparar:
- P(k) vs CLASS a múltiples redshifts.
- Halo Mass Function vs Tinker 2008.
- σ₈(z) vs teoría.
- Reproducibilidad bit-a-bit serial.

### 2.2 Configuración
Crear `configs/production_128_cubed.toml`:

```toml
[simulation]
box_size = 100.0          # Mpc/h
n_particles = 128
final_time = 1.0          # z=0
deterministic = true

[cosmology]
enabled = true
a_init = 0.01             # z=99
omega_m = 0.315
omega_b = 0.049
omega_lambda = 0.685
hubble_h = 0.674
sigma8 = 0.811
n_s = 0.965

[initial_conditions]
kind = { Zeldovich2LPT = "eisenstein_hu" }
seed = 12345

[gravity]
solver = "treepm"
theta = 0.5

[output]
snapshot_interval = 0.05  # en a, cada delta_a=0.05
out_dir = "runs/production_128"

[performance]
use_gpu = false           # CPU-only para reference bit-a-bit
n_threads = 0             # auto
```

### 2.3 Pasos concretos

```bash
# 1. Correr simulación (estimar ~2-4h en CPU moderna con 8-16 cores)
cargo build --release -p gadget-ng-cli
./target/release/gadget-ng stepping \
  --config configs/production_128_cubed.toml \
  --out runs/production_128 \
  --snapshot

# 2. Verificar que no hay warnings de drift/kick energy
#    Revisar stdout/err por líneas que contengan "energy error" o "drift"

# 3. Analizar snapshots
cargo run --release -p gadget-ng-cli -- analyze \
  --snapshot runs/production_128/snapshot_* \
  --out runs/production_128/analysis

# 4. Comparar P(k) vs CLASS
#    Usar script existente en docs/scripts/ o crear:
python3 docs/scripts/compare_pk_class.py \
  --pk runs/production_128/analysis/pk_*.txt \
  --class_transfer experiments/nbody/phase38_class_validation/reference/class_transfer.dat \
  --out runs/production_128/validation_pk.png

# 5. Comparar HMF vs Tinker
python3 docs/scripts/compare_hmf_tinker.py \
  --fof runs/production_128/analysis/fof_*.csv \
  --out runs/production_128/validation_hmf.png

# 6. Verificar reproducibilidad
#    Correr DOS VECES con la misma seed y config, comparar hashes md5 de snapshots finales.
md5sum run_a/snapshot_final run_b/snapshot_final
```

### 2.4 Criterios de aceptación
- [ ] P(k) coincide con CLASS dentro de 2% para k < k_Nyquist/2 a z=0, 1, 2.
- [ ] HMF coincide con Tinker dentro de 10% para masas > 50 partículas.
- [ ] σ₈(a) sigue curva teórica de crecimiento con error < 1%.
- [ ] Snapshots finales de dos corridas idénticas tienen mismos hashes (modo serial `deterministic = true`).
- [ ] Documentar resultado en `docs/reports/2026-05-production-128-validation.md`.

---

## 3. Optimización GPU (Prioridad: ALTA)

> **Contexto:** Los kernels existen y son correctos, pero la infraestructura per-step los mata. El objetivo de mañana no es reescribir kernels, sino **hacer la infraestructura eficiente**.

### 3.1 Problema 1: Buffer churn en wgpu (Impacto: CRÍTICO)

**Archivo:** `crates/gadget-ng-gpu/src/direct_gravity.rs` (y similares en `barnes_hut.rs`, `fmm.rs`, `treepm_shortrange.rs`)

**Síntoma:** Cada llamada a `compute_accelerations_raw` crea:
```rust
let buf_pos = ctx.device.create_buffer_init(&BufferInitDescriptor { ... });
let buf_mass = ctx.device.create_buffer_init(...);
let buf_out = ctx.device.create_buffer(...);
let buf_rb = ctx.device.create_buffer(...); // staging readback
```

**Solución:** Persistent buffers + `queue.write_buffer()` para updates.

#### Pasos concretos

- [ ] En `GpuContext` (o crear `GpuPersistentBuffers`), agregar:
  ```rust
  pub struct GpuPersistentBuffers {
      pub pos: wgpu::Buffer,
      pub mass: wgpu::Buffer,
      pub acc: wgpu::Buffer,
      pub staging: wgpu::Buffer,
      pub uniform: wgpu::Buffer,
      pub bind_group: wgpu::BindGroup,
      pub pipeline: wgpu::ComputePipeline,
      pub max_n: usize,
  }
  ```
- [ ] Implementar `GpuPersistentBuffers::new(ctx, max_n, shader_code, entry_point)` que precree todo.
- [ ] Implementar `fn upload(&self, queue, positions, masses)` que use `queue.write_buffer()` en vez de `create_buffer_init`.
- [ ] Implementar `fn dispatch_and_readback(&self, encoder, n, workgroups) -> Vec<Vec3>` que reutilice el staging buffer.
- [ ] Refactorizar `GpuDirectGravity` para que `new()` cree `GpuPersistentBuffers` y `accelerations()` solo haga `upload()` + `dispatch()` + `readback()`.
- [ ] Hacer lo mismo para `GpuBarnesHutMonopole` y `GpuBarnesHutFmm`.

**Comando para validar:**
```bash
cargo test -p gadget-ng-gpu -- --nocapture
# Verificar que tests de gravedad directa y BH siguen pasando
```

### 3.2 Problema 2: Transferencias sincrónicas (Impacto: ALTO)

**Síntoma:**
```rust
ctx.queue.submit(Some(encoder.finish()));
buf_rb.slice(..).map_async(wgpu::MapMode::Read, move |v| { tx.send(v).unwrap(); });
ctx.device.poll(wgpu::PollType::Wait);
rx.recv().expect("GPU readback failed");
```

Esto bloquea la CPU hasta que la GPU termina. No hay overlap.

**Solución:** Async pipelining con `pollster::block_on` en un hilo separado o, mejor, usar `wgpu::BufferSlice::get_mapped_range` con callback encadenada.

#### Pasos concretos

- [ ] Crear `GpuAsyncReadback` que encadene:
  1. `queue.submit()`
  2. `map_async()` con callback que copie datos a un `Arc<Mutex<Vec<Vec3>>>`
  3. `device.poll(PollType::Poll)` en un loop no bloqueante (o en thread dedicado)
- [ ] Modificar `GpuDirectGravity::accelerations()` para que devuelva `Result<Vec<Vec3>, GpuError>` pero el submit sea inmediato y el readback async.
- [ ] En `stepping.rs`, si se usa GPU, llamar a `solver.submit()` justo después del drift, y leer resultados justo antes del kick. Esto permite overlap drift CPU con gravity GPU.
  - **Archivo objetivo:** `crates/gadget-ng-cli/src/engine/stepping.rs`, función `run_stepping()`.
  - Buscar la sección donde se llama `compute_forces` y ver si se puede separar en `submit_forces()` + `await_forces()`.

### 3.3 Problema 3: Tree build en CPU cada step (Impacto: ALTO)

**Síntoma:** `GpuBarnesHutGpu::accelerations_for_indices` llama `Octree::build()` en CPU, luego exporta nodos a GPU.

**Solución:** Incremental tree update o, como MVP, batch de pasos sin rebuild.

#### Pasos concretos (MVP)

- [ ] Agregar campo `rebuild_interval: usize` a `GpuBarnesHutGpu` (default 1).
- [ ] Cachear `octree` y `gpu_nodes` como `Option<(Octree, Vec<GpuNode>)>`.
- [ ] Solo reconstruir si `step % rebuild_interval == 0`.
- [ ] Para cosmología con drift pequeño, `rebuild_interval = 5` o `10` puede ser aceptable con error controlado.
- [ ] Agregar test `bh_rebuild_interval_convergence` que compare fuerzas con `rebuild_interval=1` vs `rebuild_interval=5` para un snapshot estático y verifique error relativo < 1%.

### 3.4 Problema 4: HIP PM diverge del reference (Impacto: MEDIO)

**Archivo:** `crates/gadget-ng-hip/src/hip/pm_gravity.hip`

**Síntoma:** Usa rocFFT R2C/C2R f32 con `norm = 1/N³`, mientras CPU/CUDA usan Z2Z f64 con `1/N⁴` y `-i` explícito.

**Solución:** Unificar pipeline.

#### Pasos concretos

- [ ] Cambiar `hip/pm_gravity.hip` para usar **Z2Z f64** en vez de R2C f32.
- [ ] Verificar que rocFFT soporta `rocfft_precision_double` + `rocfft_transform_type_z2z`.
- [ ] Copiar la lógica de normalización y fase de `cuda/pm_gravity.cu` linea a linea.
- [ ] Agregar test `hip_pm_matches_cpu_fft_poisson` en `gadget-ng-hip/tests/` (similar al de CUDA).
- [ ] Si rocFFT Z2Z f64 no está disponible en la máquina de build, dejar comentario `// SAFETY: fallback to R2C f32 documented` y crear issue para trackear.

### 3.5 Problema 5: CUDA/HIP Direct Gravity no cableados al CLI (Impacto: BAJO)

**Archivo:** `crates/gadget-ng-cli/src/engine/solver_factory.rs` (o donde esté `make_solver()`)

- [ ] Buscar `SolverKind::Direct` y agregar ramas:
  ```rust
  #[cfg(feature = "cuda")]
  SolverKind::Direct if cfg.performance.use_gpu_cuda => {
      Box::new(gadget_ng_cuda::CudaDirectGravity::new(...))
  }
  #[cfg(feature = "hip")]
  SolverKind::Direct if cfg.performance.use_gpu_hip => {
      Box::new(gadget_ng_hip::HipDirectGravity::new(...))
  }
  ```
- [ ] Verificar que `CudaDirectGravity` y `HipDirectGravity` implementan el trait de solver correctamente (mismos métodos que `DirectGravity`).

### 3.6 Problema 6: Faltan benchmarks reales (Impacto: MEDIO)

- [ ] Extender `crates/gadget-ng-gpu/benches/gpu_vs_cpu.rs`:
  - Agregar grupos para `barnes_hut_monopole` y `fmm_quadrupole`.
  - Agregar N ∈ {2000, 5000, 10000, 20000}.
- [ ] Crear `crates/gadget-ng-cuda/benches/cuda_pm_vs_cpu.rs`:
  - Comparar `CudaPmSolver` vs `PmSolver` CPU para grids 64³, 128³, 256³.
- [ ] Correr benchmarks y commitear resultados en `docs/reports/2026-05-gpu-benchmarks.md`.

```bash
cargo bench -p gadget-ng-gpu --features gpu
cargo bench -p gadget-ng-cuda --features cuda
```

### 3.7 Problema 7: Tests placeholders CUDA/HIP PM (Impacto: BAJO)

**Archivo:** `crates/gadget-ng-gpu/tests/v1_gpu_tests.rs`

- [ ] Implementar `pm_gpu_roundtrip_fft`:
  - Crear grid 32³ de densidad conocida (e.g., sinusoide).
  - Correr PM GPU (CUDA si está disponible, sino wgpu si tiene PM).
  - Comparar potencial resultante con solución analítica o con CPU PM.
- [ ] Implementar `power_spectrum_pm_gpu_matches_pm_cpu`:
  - Correr ICs Zeldovich para N=32³.
  - Medir P(k) con PM CPU y PM GPU.
  - Assert relative error < 2% para k < k_Nyquist/2.

---

## 4. CI Hardening (Prioridad: MEDIA)

### 4.1 Agregar `cargo audit`

**Archivo:** `.github/workflows/ci.yml`

- [ ] Agregar job:
  ```yaml
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: rustsec/audit-check@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
  ```
- [ ] Hacerlo **advisory** (`continue-on-error: true`) la primera semana; después evaluar si se bloquea.

### 4.2 Graduar clippy-all-targets a blocking

- [ ] En `.github/workflows/ci.yml`, cambiar:
  ```yaml
  - name: Clippy all targets
    run: cargo clippy --workspace --all-targets -- -D warnings
    continue-on-error: false  # era true
  ```
- [ ] Verificar que actualmente pasa: `cargo clippy --workspace --all-targets -- -D warnings`.

### 4.3 Agregar `cargo doc` a CI

- [ ] Agregar step:
  ```yaml
  - name: Doc build
    run: cargo doc --workspace --no-deps
    env:
      RUSTDOCFLAGS: -D warnings
  ```

### 4.4 Agregar MSRV a `scripts/check.sh`

**Archivo:** `scripts/check.sh`

- [ ] Agregar después de `cargo fmt`:
  ```bash
  echo "=== MSRV check ==="
  rustup run 1.85 cargo check --workspace
  ```
  (o usar `cargo msrv verify` si se instala `cargo-msrv`).

### 4.5 Expandir MPI multirank

**Archivo:** `.github/workflows/ci.yml`, job `mpi-multirank`

- [ ] Agregar tests adicionales:
  ```bash
  mpirun -n 4 ./target/debug/deps/treepm_distributed-*
  mpirun -n 2 ./target/debug/deps/v2_hierarchical_cosmo-*
  mpirun -n 4 ./target/debug/deps/phase43_dt_treepm_parallel-*
  ```
- [ ] Asegurar que estos tests no cuelguen en CI (algunos pueden necesitar timeout o `#[ignore]` si usan MPI real).

---

## 5. Física Nueva (Prioridad: BAJA / POST-JOSS)

> **Decisión ejecutiva:** No agregar física nueva hasta que JOSS y validación N=128³ estén cerrados.

### 5.1 Post-JOSS: Roadmap de física

| Rank | Física | Justificación | Complejidad |
|------|--------|---------------|-------------|
| 1 | **Neutrinos por partículas** | Diferenciador vs GADGET-4. Impacto en P(k) a small scales. | Alto |
| 2 | **RT multi-frecuencia** | Necesario para reionización realista. M1 gray es oversimplified. | Medio-Alto |
| 3 | **Zoom / particle splitting** | Permite simulaciones de galaxias individuales con resolución variable. | Alto |
| 4 | **MHD no-ideal** | Ambipolar + Ohmic. Importante para formación estelar. | Medio |
| 5 | **f_NL en ICs** | Bajo esfuerzo, alto valor para comunidad CMB-LSS. | Bajo |
| 6 | **Small-scale dynamo** | Subgrid para amplificación de B. | Medio |

### 5.2 Física que NO hace falta (explicado)

- **DGP / symmetron / scalar-tensor:** f(R) Hu-Sawicki ya cubre modified gravity screening. Agregar más es law of diminishing returns.
- **SIDM con drag / inelastic:** Elástico isotrópico es el benchmark. Drag requiere modelo de dark sector específico.
- **Ram-pressure stripping explícito:** En SPH ocurre implícitamente por hydrodinámica. Modelos explícitos son subgrid ad-hoc.
- **BAO reconstruction:** Es análisis post-proceso, no física del solver.

---

## 6. Runbooks Faltantes (Prioridad: BAJA)

Crear estos archivos en `docs/runbooks/`:

### 6.1 `gpu-setup.md`
- Requisitos: Vulkan drivers, CUDA toolkit, ROCm.
- Features de cargo: `--features gpu`, `--features cuda`, `--features hip`.
- Troubleshooting: `wgpu` no detecta adapter, `nvcc` no encontrado, `hipcc` no encontrado.
- Validación mínima: `cargo test -p gadget-ng-gpu -- --nocapture`.

### 6.2 `debugging-mpi-hangs.md`
- Síntoma: `cargo test --workspace` cuelga.
- Causa: tests MPI que necesitan `mpirun` pero corren sin él.
- Fix: correr tests MPI individualmente con `mpirun -n 2`.
- Debug: `MPI_DEBUG=1`, `RUST_LOG=debug`, `timeout 60`.

### 6.3 `release-process.md`
- Version bump en `Cargo.toml` workspace.
- Update `CHANGELOG.md`.
- Tagging: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
- Build release: `bash scripts/check_release.sh`.
- GitHub Release + Zenodo DOI.

---

## 7. Orden de ejecución recomendado para mañana

Si tenés 8 horas, distribuí así:

| Horario | Tarea | Archivos clave |
|---------|-------|----------------|
| 0:00–1:00 | **JOSS metadatos + abstract** | `docs/paper/paper.md` |
| 1:00–2:30 | **Figuras paper** | `docs/paper/figures/generate_all.py`, scripts Python existentes |
| 2:30–3:00 | **Compilar PDF JOSS** | `docs/paper/Makefile`, pandoc |
| 3:00–4:00 | **Validación N=128³: config + launch** | `configs/production_128_cubed.toml`, `scripts/check-physics.sh` |
| 4:00–5:00 | **GPU persistent buffers (wgpu)** | `crates/gadget-ng-gpu/src/direct_gravity.rs`, `GpuPersistentBuffers` |
| 5:00–6:00 | **GPU async readback** | `crates/gadget-ng-gpu/src/lib.rs`, `stepping.rs` |
| 6:00–7:00 | **CI: audit + clippy blocking + doc** | `.github/workflows/ci.yml` |
| 7:00–8:00 | **Buffer / commit / push** | `git add`, `git commit`, `scripts/check.sh` |

**Si solo tenés 4 horas:** Hacé solo JOSS (metadatos + figuras + PDF) + validación N=128³ config. El GPU y CI pueden esperar al día siguiente.

---

## 8. Criterios de cierre del día

Antes de hacer `git push`, verificar:

```bash
# Tests
bash scripts/check-physics.sh
cargo test --workspace

# Format + clippy
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo clippy --workspace --all-targets -- -D warnings

# Docs
cargo doc --workspace --no-deps

# Status
git diff --stat
```

- [ ] Todo pasa.
- [ ] Nuevos archivos (`configs/`, `docs/paper/figures/`, `docs/runbooks/`) están trackeados.
- [ ] Commit con mensaje apropiado:
  ```
  feat(docs): plan JOSS y validación producción N=128³
  
  - Completa metadatos y figuras del paper JOSS
  - Agrega config production_128_cubed.toml
  - Refactoriza GPU persistent buffers en wgpu
  - Hardening CI: audit, clippy blocking, doc build
  ```

---

> **Nota para Cristian:** Este documento es un plan, no un mandato. Si algo toma más tiempo del esperado, creá un issue en GitHub para trackearlo y seguí con lo siguiente. No te quedes atascado en un solo punto.
