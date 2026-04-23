# Phase 71 — Bispectrum B(k₁,k₂,k₃)

**Fecha**: Abril 2026  
**Crate**: `gadget-ng-analysis`  
**Módulo**: `src/bispectrum.rs`

---

## Motivación

El bispectrum es la estadística de 3 puntos en espacio de Fourier y el principal detector de
no-gaussianidades en la distribución de materia. Mientras que el espectro de potencia P(k)
captura toda la información de un campo gaussiano, los modos gravitacionalmente acoplados
generan un B(k₁,k₂,k₃) ≠ 0. Su medición permite:

- Detectar acoplamiento no-lineal gravitacional.
- Constrainar la no-gaussianidad primordial f_NL.
- Validar simulaciones contra teoría de perturbaciones de un lazo.

---

## Implementación

### Algoritmo (shell-filter)

1. **CIC deposit** → δ(x) en grid N³.
2. **FFT 3D** (3× 1D, rustfft) → δ(k).
3. Para cada bin k:
   - Filtrar modos en la cáscara |k| ∈ [k−Δk/2, k+Δk/2].
   - IFFT → δ_k(x) en espacio real.
4. **Bispectrum equiláteral**: B_eq(k) = ⟨δ_k³(x)⟩ × V².
5. **Bispectrum isósceles**: B(k₁,k₂) = ⟨δ_k₁(x) δ_k₂(x) δ_{k₃}(x)⟩ × V².
6. **Bispectrum reducido**: Q(k) = B_eq(k) / [3 × P(k)²].

### Complejidad

| Variante       | Costo              |
|----------------|--------------------|
| Equiláteral    | O(n_bins × N³ log N) |
| Isósceles      | O(n_bins² × N³ log N) |

---

## API Pública

```rust
// Bispectrum equiláteral B_eq(k)
pub fn bispectrum_equilateral(
    positions: &[Vec3],
    masses:    &[f64],
    box_size:  f64,
    mesh:      usize,
    n_bins:    usize,
) -> Vec<BkBin>

// Bispectrum isósceles B(k₁, k₂)
pub fn bispectrum_isosceles(
    positions: &[Vec3],
    masses:    &[f64],
    box_size:  f64,
    mesh:      usize,
    k1_bins:   &[f64],
    k2_bins:   &[f64],
) -> Vec<BkIsoscelesBin>

// Bispectrum reducido Q(k) = B_eq / 3P²
pub fn reduced_bispectrum(
    bk_bins: &[BkBin],
    pk_table: &[(f64, f64)],
) -> Vec<(f64, f64)>
```

### Structs

```rust
pub struct BkBin {
    pub k: f64,
    pub bk: f64,
    pub n_triangles: u64,
}

pub struct BkIsoscelesBin {
    pub k1: f64, pub k2: f64,
    pub bk: f64,
    pub n_triangles: u64,
}
```

Ambas derivan `Serialize + Deserialize`.

---

## Tests

| Test                              | Verificación                                   |
|-----------------------------------|------------------------------------------------|
| `bk_uniform_nearly_zero`          | Para red uniforme, B_eq ≈ 0                   |
| `bk_bins_have_k_increasing`       | Bins en orden creciente de k                   |
| `bk_finite_for_random_distribution` | B_eq finito para distribución aleatoria       |
| `bk_bin_struct_serializes`        | Round-trip JSON de `BkBin`                    |
| `reduced_bispectrum_gaussian_near_zero` | Q finito para campo gaussiano aprox.   |

---

## Referencia

- Scoccimarro (2000), ApJ 544, 597.  
- Sefusatti & Komatsu (2007), PRD 76, 083004.  
- Verde et al. (2002), MNRAS 335, 432.
