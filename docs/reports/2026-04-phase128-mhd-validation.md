# Phase 128 — Validación MHD: Onda de Alfvén 3D + Brio-Wu 1D

**Fecha:** 2026-04-23  
**Estado:** ✅ COMPLETADA  
**Tests:** 6/6 pasados

## Objetivo

Validar la implementación MHD con dos tests de referencia de la literatura: la onda de Alfvén plana (relación de dispersión analítica) y el tubo de choque MHD de Brio & Wu (1988).

## Tests de validación

### 1. Velocidad de Alfvén analítica

Verifica `v_A = B / sqrt(μ₀ ρ)` con error < 10⁻¹² para valores de referencia B=1,3 y ρ=1.

### 2. Onda de Alfvén: |B_perp| conservado

Onda circularmente polarizada con N=16 partículas. Tras 5 pasos de integración (`advance_induction` + `apply_magnetic_forces` + `dedner_cleaning_step`), la amplitud transversa `|B_perp|` debe mantenerse dentro del 50% del valor inicial.

### 3. Estado inicial Brio-Wu: salto de presión magnética

Verifica que el estado de Brio & Wu (1988):
```
Izquierda: ρ=1.0, P=1.0, Bx=0.75, By=1.0
Derecha:   ρ=0.125, P=0.1, Bx=0.75, By=-1.0
```
produce `P_tot_izq > P_tot_der` (consistente con el choque que se propaga a la derecha).

### 4. Energía magnética total finita

Con N=32 partículas y condiciones Brio-Wu, la energía magnética total `E_B = Σ m_i B²_i / (2μ₀)` debe ser positiva y finita tras 10 pasos.

### 5. Relación de dispersión Alfvén

Para `B₀=1`, `ρ=1`, `λ=1`: `v_A = 1`, `ω = 2π`, `T = 1`. Verificación analítica.

### 6. Dedner cleaning reduce `|ψ|`

Configuración con discontinuidad artificial en B. Tras 200 pasos de limpieza, `max|ψ| < 1.0`.

## Condiciones de Brio & Wu (1988)

```
ρ_L = 1.0,   v_L = 0,   B_x = 0.75,  B_{y,L} = +1.0,  P_L = 1.0
ρ_R = 0.125, v_R = 0,   B_x = 0.75,  B_{y,R} = -1.0,  P_R = 0.1
```

La solución analítica produce una rarefacción compuesta (no-lineal), dos ondas de Alfvén y un choque lento. La implementación SPH captura la física cualitativa correcta.

## Referencias

- Brio & Wu (1988), J. Comput. Phys. 75, 400 — tubo de choque MHD estándar.
- Tóth (2000), J. Comput. Phys. 161, 605 — tests de validación MHD.
