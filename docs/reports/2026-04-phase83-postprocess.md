# Phase 83 — Post-procesamiento automático + README

**Fecha**: 2026-04-23  
**Crates**: `docs/notebooks/`, `README.md`  
**Tipo**: Documentación / herramientas

---

## Resumen

Script Python de post-procesamiento automático para los archivos `insitu_*.json`,
y actualización completa del README con secciones para Phases 71–83.

---

## 83a — Script `postprocess_insitu.py`

**Archivo**: `docs/notebooks/postprocess_insitu.py`

### Funcionalidades

| Función | Descripción |
|---------|-------------|
| `load_insitu_dir(dir)` | Carga todos los `insitu_*.json` ordenados por paso |
| `sigma8_from_pk(pk, box)` | Integra P(k) con ventana top-hat R=8 Mpc/h |
| `plot_pk_evolution(...)` | P(k) para 8 redshifts representativos (matplotlib) |
| `plot_pk_multipoles(...)` | P₀/P₂/P₄ al snapshot final |
| `plot_sigma8_evolution(...)` | σ₈(z) como función del redshift |
| `plot_nhalos_evolution(...)` | n_halos(z) y masa total en halos |
| `plot_bispectrum(...)` | B_eq(k) al último snapshot |
| `build_summary(...)` | JSON con series temporales de todos los estadísticos |

### Dependencias opcionales

- `numpy` — integración σ₈, manipulación de arrays
- `matplotlib` — gráficos (se omiten si no disponible)
- `scipy` — integración de Simpson (usa trapz si no disponible)

### Uso

```bash
python docs/notebooks/postprocess_insitu.py \
    --dir runs/cosmo/insitu \
    --out analysis/ \
    --box-size 100.0
```

**Salidas**:
- `analysis/pk_evolution.png` — P(k) en varios redshifts
- `analysis/pk_multipoles.png` — multipoles RSD
- `analysis/sigma8_evolution.png` — σ₈(z)
- `analysis/halos_evolution.png` — n_halos y M_total en halos
- `analysis/bispectrum.png` — B_eq(k) a z=0
- `analysis/summary.json` — series temporales completas

## 83b — Actualización README

Añadida la sección **"Estadísticas avanzadas y transferencia radiativa (Phases 71–82)"** con:
- Documentación de cada phase (71–83): descripción, ejemplo de uso, TOML
- Tabla de hitos actualizada con entries para Phases 61–83
- Crates actualizados: `gadget-ng-sph` con feedback, `gadget-ng-rt` con MPI y química

---

## Tests

- Script se ejecuta sin errores (requiere Python ≥ 3.10).
- Build completo del workspace: ✅
