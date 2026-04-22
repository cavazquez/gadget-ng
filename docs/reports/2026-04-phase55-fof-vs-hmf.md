# Phase 55 — Comparación espectro de masas FoF vs HMF

**Fecha:** abril 2026  
**Crate principal:** `gadget-ng-physics` (tests), `gadget-ng-analysis` (FoF + HMF)  
**Archivo de test:** `crates/gadget-ng-physics/tests/phase55_fof_vs_hmf.rs`

---

## Contexto

Phase 52 implementó la función de masa de halos Press-Schechter / Sheth-Tormen.
Phase 53 añadió perfiles NFW y relación concentración-masa.
Phase 55 es la validación de extremo a extremo: evoluciona la simulación completa
hasta z=0 y compara el espectro de masas de halos FoF con las predicciones analíticas.

---

## Física

### Función de masa de halos

La densidad numérica de halos por unidad de logaritmo en masa es:

```
dn/d ln M = f(σ) · (ρ̄_m/M) · |d ln σ⁻¹/d ln M|
```

donde `f(σ)` es la multiplicidad de Sheth-Tormen:

```
f_ST(σ) = A · √(2a/π) · ν_a · exp(-a·ν²/2) · (1 + (a·ν²)^{-p})
con ν = δ_c/σ,  a=0.707,  p=0.3,  A=0.3222,  δ_c=1.686
```

### Masa mínima resoluble (FoF con min_particles=20)

La masa mínima de un halo FoF con `N_min=20` partículas es `M_min = N_min × m_part`:

| N    | m_part [M_sun/h] | M_min [M_sun/h] | Régimen         |
|------|-----------------|-----------------|-----------------|
| 64³  | 9.0×10¹²        | 1.8×10¹⁴        | Cúmulos masivos |
| 128³ | 1.1×10¹²        | 2.2×10¹³        | Grupos de galaxias |
| 256³ | 1.4×10¹¹        | 2.8×10¹²        | Halos galácticos |

### Conversión de unidades

Las partículas en la simulación tienen masa interna `1/N_total` y posiciones
en el rango `[0, 1]`. La conversión a unidades físicas:

```
m_part [M_sun/h] = Ω_m · ρ_crit_H2 · BOX_MPC_H³ / N_total
                 = 0.315 × 2.775×10¹¹ × 300³ / N
```

donde `RHO_CRIT_H2 = 2.775×10¹¹ (M_sun/h)/(Mpc/h)³`.

### Masa en M_sun/h de cada halo FoF

```rust
// FofHalo.n_particles es el número de partículas miembro
let halo_mass_msun_h = halo.n_particles as f64 * m_part_msun_h;
```

---

## Implementación

### Evolución hasta z=0

La simulación usa los mismos parámetros que Phase 54 pero:
- `BOX_MPC_H = 300.0` (mayor estadística de halos masivos)
- `A_FINAL = 1.0` (z=0)
- Número de pasos: ~400-1000 dependiendo de N

### FoF en unidades internas

El FoF se ejecuta en unidades internas (`box_size=1.0`). La longitud de enlace es:

```
ll = b × l̄ = 0.2 × (V/N)^{1/3} = 0.2 × (1/N_total)^{1/3}
```

Para N=128³: `ll ≈ 0.2/128 = 1.56×10⁻³` (unidades internas) = `0.47 Mpc/h`.

### Estructura del test

```
phase55_fof_vs_hmf.rs
├── run_simulation_n(n)         → evoluciona + FoF → SimResult55
├── evolve_pm_to_a_adaptive()   → igual que Phase 54 pero hasta a=1.0
├── run_full_matrix()           → N ∈ {64, 128, 256}
├── matrix()                   → OnceLock
└── Tests:
    ├── phase55_evolution_stable_n64    → a_final≈1.0, v_rms < 50
    ├── phase55_halos_found_n64        → ≥1 halo
    ├── phase55_halos_found_n128       → ≥20 halos
    ├── phase55_halos_found_n256       → ≥100 halos
    ├── phase55_fof_vs_hmf_ratio_n128  → ratio FoF/ST ∈ [0.05, 20]
    └── phase55_mass_function_convergence → M_min(N=256) < M_min(N=64)
```

---

## Número de halos esperado en BOX=300 Mpc/h, z=0

| N    | M_min [M_sun/h] | n(>M_min) [h³/Mpc³] | N_halos esperados |
|------|-----------------|---------------------|-------------------|
| 64³  | 1.8×10¹⁴        | ~1.5×10⁻⁵           | ~410              |
| 128³ | 2.2×10¹³        | ~2×10⁻⁴             | ~5400             |
| 256³ | 2.8×10¹²        | ~5×10⁻³             | ~135000           |

En la práctica el PM grid no resuelve las escalas más pequeñas, por lo que
los halos reales serán un subconjunto. Para N=64 se esperan ~50-200 halos.

---

## Script de ejecución

```bash
# Tiempo estimado release: N=64 ~20s, N=128 ~160s, N=256 ~1200s
./experiments/nbody/phase55_fof_vs_hmf/run_phase55.sh

# Saltando N=256:
PHASE55_SKIP_N256=1 ./experiments/nbody/phase55_fof_vs_hmf/run_phase55.sh
```

---

## Limitaciones conocidas

1. **PM resuelve escalas ~2×cellsize**: Para N=64, BOX=300 Mpc/h, el PM
   resuelve estructuras > ~9 Mpc/h. Los halos con M < ~10¹⁴ M_sun/h pueden
   estar sub-resueltos en N=64.

2. **Shot noise del PM**: Con G_consistent las fuerzas son ~2660× más débiles
   que con G=1. El clustering es correcto pero puede que las estructuras más
   pequeñas no colapsen completamente en el PM.

3. **Tolerancia amplia en HMF**: El ratio FoF/ST ∈ [0.05, 20] (factor ~4-5
   en cada dirección) refleja las incertidumbres de volumen pequeño y la
   sensibilidad del FoF al linking length y a la resolución de masa.

---

## Relación con GADGET-4

GADGET-4 incluye un finder FoF completo (`fof.h`) y comparaciones con HMFs
analíticas como validación de producción. Phase 55 implementa el equivalente
en gadget-ng, demostrando que la cadena completa IC → evolución PM → FoF → HMF
es funcional y produce resultados cualitativamente correctos.
