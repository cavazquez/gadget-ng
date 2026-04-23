# Validaciones físicas pendientes

**Fecha:** 2026-04-23

Este documento registra las validaciones físicas de mayor envergadura que requieren
tiempo de cómputo o entorno MPI completo, para realizarse en sesiones futuras.

---

## 1. Test EoR cosmológico real (z=12→6)

**Estimación:** 2–3 sesiones

**Objetivo:** Verificar que el frente de ionización se propaga correctamente en
una simulación cosmológica real con SPH + RT M1 + reionización acoplada.

**Configuración sugerida:**

```toml
# configs/eor_cosmo_real.toml
[simulation]
particle_count = 32768   # 32³, ~manageable en 1 hora
box_size       = 20.0    # Mpc/h

[cosmology]
enabled = true
a_init  = 0.0769   # z = 12
a_end   = 0.143    # z = 6

[sph]
enabled = true
gamma   = 5.0
n_neigh = 32

[rt]
enabled      = true
rt_mesh      = 16
c_red_factor = 100.0
substeps     = 3

[reionization]
enabled       = true
n_sources     = 8
uv_luminosity = 1.0
z_start       = 12.0
z_end         = 6.0

[insitu_analysis]
enabled          = true
interval         = 50
cm21_enabled     = true
igm_temp_enabled = true
pk_mesh          = 16
```

**Verificaciones:**
- `x_hii_mean` sube de ~0 a ~1 entre z=12 y z=6
- P(k)₂₁cm muestra pico y caída a medida que avanza la reionización
- T_IGM(z) sube de ~100 K a ~10⁴ K en el gas ionizado
- R_Strömgren numérico ≈ analítico para fuentes puntuales

**Cómo ejecutar:**
```bash
cargo run --release -p gadget-ng-cli -- run configs/eor_cosmo_real.toml
python docs/notebooks/postprocess_insitu.py runs/eor_cosmo_real/insitu/
```

---

## 2. AGN con halos FoF identificados in-situ

**Estimación:** 1 sesión

**Objetivo:** Colocar los agujeros negros semilla en los centros de los halos FoF
más masivos en lugar de en posiciones fijas, lo que requiere:

1. En el análisis in-situ, ordenar los halos FoF por masa y extraer los N_bh
   centros de masa más masivos.
2. En `maybe_agn!`, actualizar las posiciones de `agn_bhs` con los centros de halos.

**Cambios necesarios:**
- `maybe_run_insitu` debe retornar los centros de halos si `sph.agn.enabled`
- `maybe_agn!` recibe la lista de centros y actualiza `agn_bhs[i].pos`
- Nuevo campo `halo_centers: Option<Vec<[f64; 3]>>` en el retorno de `maybe_run_insitu`
  o como variable compartida en el engine

**Test:** Verificar que `agn_bhs[i].pos` queda dentro del radio virial del halo i
tras la actualización.

---

## 3. Producción N=128³ completa hasta z=0

**Estimación:** 2–3 sesiones (incluyendo tiempo de cómputo: ~2–4 horas MPI)

**Objetivo:** Corrida de producción completa para validación formal del código:
- TreePM + SPH + feedback SN
- P(k) a z={4,2,1,0.5,0} comparado con HaloFit (Takahashi+2012)
- HMF a z={1,0.5,0} comparado con Tinker (2008)
- c(M) de halos FoF comparado con Duffy+2008

**Configuración:** `configs/validation_128.toml` (ya existe, Phase 79)

**Script MPI sugerido:**
```bash
mpirun -np 8 cargo run --release -p gadget-ng-cli -- run configs/validation_128.toml
python docs/notebooks/postprocess_insitu.py runs/validation_128/insitu/
python docs/notebooks/bench_pk_vs_gadget4.py runs/validation_128/insitu/
```

**Métricas de éxito:**
- P(k, z=0): error RMS < 5% respecto a HaloFit para k < k_Nyquist/2
- HMF, z=0: ratio FoF/Tinker entre 0.7 y 1.5 para M ∈ [10¹², 10¹⁵] M_sol/h
- σ₈(z=0) reproducido con error < 2%

---

## 4. Submission JOSS

**Estimación:** 1 sesión

**Pasos:**
1. `pip install matplotlib numpy && python docs/notebooks/generate_paper_figures.py`
2. Completar ORCIDs y afiliaciones en `docs/paper/paper.md`
3. Crear release `v0.1.0` en GitHub y obtener DOI Zenodo
4. Compilar PDF: `docker run openjournals/inara -o paper.pdf paper.md`
5. Pre-submission inquiry en https://joss.theoj.org/papers/new

**Checklist completo:** `docs/paper/submission_checklist.md`

---

## Resumen de prioridades

| Tarea | Prioridad | Dependencias | Estimación |
|-------|-----------|--------------|-----------|
| Submission JOSS | Alta | Figuras generadas, DOI Zenodo | 1 sesión |
| Test EoR cosmológico | Media | MPI disponible, ~1h cómputo | 2–3 sesiones |
| AGN con halos FoF | Media | maybe_run_insitu refactorizado | 1 sesión |
| Producción N=128³ | Media-baja | MPI, ~4h cómputo | 2–3 sesiones |
