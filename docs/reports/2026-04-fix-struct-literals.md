# Fix — Actualización de literales de Particle y RunConfig

**Fecha:** abril 2026  
**Tipo:** Mantenimiento / corrección de compilación  
**Commit:** `fix: actualizar literales de Particle y RunConfig tras campos SPH/insitu`

---

## Contexto

Las fases G2 (SPH cosmológico) y Phase 63 (in-situ analysis) agregaron nuevos
campos a dos structs fundamentales del workspace:

| Struct | Campos nuevos | Fase |
|--------|--------------|------|
| `gadget_ng_core::Particle` | `ptype: ParticleType`, `internal_energy: f64`, `smoothing_length: f64` | G2 |
| `gadget_ng_core::RunConfig` | `insitu_analysis: InsituAnalysisSection`, `sph: SphSection` | Phase 63 / G2 |

Ambos structs usan `#[serde(default)]` para compatibilidad de serialización
(TOML, JSON), pero Rust **exige** que los inicializadores de struct literal
incluyan todos los campos. Esto causó errores de compilación en **55 archivos**
de tests distribuidos a lo largo del workspace.

---

## Archivos afectados

### Tests con `Particle { ... }` literal → reemplazado por `Particle::new(...)`

| Archivo | Descripción |
|---------|-------------|
| `crates/gadget-ng-treepm/tests/minimum_image.rs` | Helper `make_particle` |
| `crates/gadget-ng-treepm/tests/sr_sfc_geometry.rs` | Helper `make_particle` |
| `crates/gadget-ng-treepm/tests/pm_scatter_gather.rs` | Helper `make_particle` |
| `crates/gadget-ng-treepm/tests/halo3d.rs` | Helper `make_particle` |
| `crates/gadget-ng-physics/tests/treepm_halo3d.rs` | Helper `make_particle` |
| `crates/gadget-ng-physics/tests/treepm_distributed.rs` | Helper `make_particle` |
| `crates/gadget-ng-physics/tests/treepm_pm_sg.rs` | Helper `make_particle` |
| `crates/gadget-ng-physics/tests/treepm_sr_sfc.rs` | Helper `make_particle` |
| `crates/gadget-ng-physics/tests/cosmo_pm_dist.rs` | Literal inline en map |
| `crates/gadget-ng-treepm/src/distributed.rs` | Helper en `#[cfg(test)]` |

### Tests con `RunConfig { ... }` literal → agregado `insitu_analysis` y `sph`

37 archivos de tests en:
- `crates/gadget-ng-tree/tests/`
- `crates/gadget-ng-treepm/tests/`
- `crates/gadget-ng-core/tests/`
- `crates/gadget-ng-integrators/tests/`
- `crates/gadget-ng-parallel/tests/`
- `crates/gadget-ng-physics/tests/`

---

## Solución aplicada

### Para `Particle`

Reemplazar el literal por el constructor seguro:

```rust
// Antes
Particle {
    position: Vec3::new(x, y, z),
    velocity: Vec3::zero(),
    acceleration: Vec3::zero(),
    mass,
    global_id: id,
}

// Después
Particle::new(id, mass, Vec3::new(x, y, z), Vec3::zero())
```

El constructor `Particle::new` inicializa los nuevos campos con sus defaults
(`ParticleType::DarkMatter`, `0.0`, `0.0`), que es el comportamiento correcto
para tests de partículas de materia oscura.

### Para `RunConfig`

Agregar los dos campos faltantes con `Default::default()`:

```rust
RunConfig {
    // ... campos existentes ...
    decomposition: Default::default(),
    insitu_analysis: Default::default(),   // ← nuevo
    sph: Default::default(),               // ← nuevo
}
```

---

## Impacto

- **0 errores de compilación** tras el fix.
- **0 cambios de comportamiento**: los nuevos campos tienen valores default
  que preservan la semántica original de todos los tests.
- **Todos los tests existentes siguen pasando** sin modificación de lógica.

---

## Lección aprendida

Al agregar campos a structs del workspace usados en tests como struct literals,
es necesario actualizar simultáneamente todos los puntos de inicialización.
Una estrategia alternativa para el futuro es usar `..Default::default()` al
final de los literales, lo que requiere que el struct implemente `Default`.
Para `RunConfig` esto no es trivial porque `SimulationSection` e
`InitialConditionsSection` tienen campos requeridos sin default semántico claro.

**Recomendación:** preferir siempre constructores o builder patterns sobre
literales de struct para tipos del núcleo (`Particle`, `RunConfig`) que
evolucionan con nuevas fases.
