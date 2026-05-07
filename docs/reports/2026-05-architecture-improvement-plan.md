# Plan de Mejora de Arquitectura — `gadget-ng`

> **Auditoría:** 2026-05-06 | **Resultado global:** 4.2/5
> **Crates:** 18 | **Tests:** 1,334 | **Scripts:** 64

---

## Fase 0 — Pre-flight: Quick Wins (< 2h)

| # | Tarea | Archivo | Estado |
|---|-------|---------|--------|
| 0.1 | Criterion workspace consistency | `crates/gadget-ng-mhd/Cargo.toml` | ✅ |
| 0.2 | `rustfmt.toml` edition 2021→2024 | `rustfmt.toml` | ✅ |
| 0.3 | `eval` inseguro → array expansion | `scripts/run_all_validations.sh` | ✅ |
| 0.4 | `set -euo pipefail` faltante | `experiments/nbody/phase4/.../run_all_tests.sh` | ✅ |
| 0.5 | Heredoc bug (variable expansion) | `scripts/bench_mpi_scaling.sh` | ✅ |
| 0.6 | `cargo fmt --all` → `--check` | `scripts/check_release.sh` | ✅ |
| 0.7 | AGENTS.md 22→18 crates + gpu-layout | `AGENTS.md` | ✅ |

---

## Fase 1 — Build & Performance Foundation (2-3 días)

### 1.1 Perfil de release optimizado para HPC

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = "symbols"
panic = "abort"

[profile.bench]
lto = "thin"
codegen-units = 1
```

### 1.2 Toolchain pinning

```toml
[toolchain]
channel = "stable"
components = ["clippy", "rustfmt"]
targets = ["x86_64-unknown-linux-gnu"]
```

### 1.3 Workspace deps — consolidar `approx`, `tempfile`

---

## Fase 2 — Safety & Code Quality (3-5 días)

- Documentar ~100 bloques `unsafe` con `// SAFETY:`
- Migrar 44 `#[allow(clippy::...)]` → `#[expect(clippy::...)]`
- Reemplazar `.unwrap()` sin mensaje en producción por `.expect()`

---

## Fase 3 — Structural Refactor (1-2 semanas)

- Descomponer `stepping.rs` (3,077 líneas → ~6 módulos)
- Limpiar `main.rs` (duplicación Analyze/Analyse, renderizado inline)

---

## Fase 4 — Testing & CI Hardening (1-2 semanas)

- Tests unitarios en `gadget-ng-mhd`
- Test helpers compartidos en `gadget-ng-physics`
- Adoptar `approx` en todo el workspace
- `proptest` para funciones numéricas core
- Golden-file tests para reproducibilidad
- CI: `cargo-audit`, `cargo-deny`, feature matrix, `shellcheck`, benchmark regression

---

## Fase 5 — Polish & Documentation (1 semana)

- Crate-level docs (`//!`) y `#![warn(missing_docs)]`
- `CONTRIBUTING.md`, `SECURITY.md`
- Seguridad de scripts (`trap`, `mktemp`, sin `git rev-parse`)

---

> **Dependencias:** Fase 0 → 1 → 2 → 3 → 4/5 (paralelas)
> **Tiempo total estimado:** 4-6 semanas
