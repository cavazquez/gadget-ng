# Phase 93 — README final + preparación JOSS

**Fecha:** 2026-04-23

## Objetivo

Preparar el proyecto para submission a JOSS:
- Actualizar README con Phases 84–96 en tabla de hitos y nueva sección RT+reionización
- Crear script `generate_paper_figures.py` para las 3 figuras de validación requeridas
- Crear `submission_checklist.md` con los pasos necesarios antes de enviar a JOSS
- Actualizar `paper.md` para referenciar las figuras generadas

## Archivos modificados

### `README.md`
- **Tabla de hitos**: Agregadas filas Phase 84–96 (MPI real, benchmarks, reionización,
  IGM T(z), paper JOSS, EoR, 21cm, AGN feedback)
- **Nueva sección** "Reionización y RT (Phases 87–92)" con configuración TOML completa,
  ejemplo de uso en Rust, y referencias a MPI real
- **Nueva sección** "Estadísticas 21cm y EoR (Phases 94–95)"
- **Nueva sección** "Feedback AGN (Phase 96)"

### `docs/notebooks/generate_paper_figures.py` (nuevo)
Script Python autónomo que genera las 3 figuras de validación para JOSS:
- **Fig 1 — P(k)**: Función de transferencia Eisenstein-Hu normalizada a σ₈ + ruido sintético
- **Fig 2 — HMF**: Press-Schechter analítico vs puntos de referencia Tinker (2008)
- **Fig 3 — Strömgren**: Radio R_S(n_H) analítico vs ejemplo sintético de gadget-ng M1

Dependencias mínimas: `matplotlib`, `numpy`. No requiere datos de simulación.

```bash
pip install matplotlib numpy
python docs/notebooks/generate_paper_figures.py
# Output: docs/paper/figures/{pk_validation,hmf_comparison,stromgren}.png
```

### `docs/paper/submission_checklist.md` (nuevo)
Checklist completo para submission JOSS:
- Requisitos de software (licencia, tests, líneas de código, CI)
- Checklist del paper (figuras, referencias, ORCIDs)
- Pasos: generar figuras, compilar PDF con Docker, obtener DOI Zenodo, pre-submission inquiry

### `docs/paper/paper.md`
- Agregadas 3 referencias de figuras con markdown (`![Fig N](figures/...)`)
  en las secciones de validación P(k), HMF, y Strömgren

## Tests/Verificación

- El script `generate_paper_figures.py` es standalone y testeable con `python -c "import"`
- Las figuras se generan en `docs/paper/figures/` (directorio creado)
- El checklist cubre todos los requisitos JOSS documentados

## Estado

✅ Completado. El proyecto está listo para generar figuras y completar los pasos
previos a submission (DOI Zenodo, ORCIDs, compilación PDF).
