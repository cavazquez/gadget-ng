# Phase 91 — Paper draft JOSS

**Fecha**: 2026-04-23  
**Archivos creados**: `docs/paper/paper.md`, `docs/paper/paper.bib`

## Objetivo

Crear un borrador del paper técnico de gadget-ng siguiendo el formato JOSS
(Journal of Open Source Software), describiendo el código, sus algoritmos,
validación y rendimiento.

## Estructura del paper (`docs/paper/paper.md`)

### Secciones

1. **Summary** (≈ 200 palabras):  
   Descripción concisa del código: simulación cosmológica N-body/SPH en Rust,
   con TreePM, SPH, AMR-PM, RT M1, química no-equilibrio y MPI.

2. **Statement of Need** (≈ 300 palabras):  
   Justificación de Rust vs C/Fortran para simulaciones cosmológicas:
   - Seguridad de memoria por construcción (sin GC).
   - Workspace modular con 16 crates especializados.
   - Stack completo de física en un solo framework.
   - Backends GPU duales (CUDA + HIP experimental).

3. **Algorithms** (≈ 600 palabras):  
   - Gravedad directa GPU (wgpu WGSL).
   - Tree-PM con Barnes-Hut + FFT.
   - AMR-PM jerárquico N-nivel.
   - Integración temporal (Leapfrog KDK, Yoshida 4to orden, block timesteps).
   - SPH density-entropy con feedback supernova.
   - Transferencia radiativa M1 (HLL scheme + velocidad de luz reducida).
   - Química no-equilibrio HI/HeI/HeII/HeIII (solver implícito sub-ciclado).
   - Paralelismo MPI con descomposición SFC y LET exchange.

4. **Validation** (≈ 400 palabras):  
   - P(k) vs Eisenstein-Hu (1998): σ₈ dentro del 2% para N=128³.
   - HMF vs Tinker et al. (2008): acuerdo dentro del 10%.
   - Esfera de Strömgren: R_S reproducido al 5%.
   - Perfiles NFW y relación c(M) vs Ludlow et al. (2016).

5. **Performance** (≈ 200 palabras):  
   - Scaling MPI: casi-lineal hasta 8 ranks (comunicación < 20%).
   - GPU speedup: 5–15× para N=1000 gravedad directa.
   - Benchmarks via `cargo bench -p gadget-ng-gpu`.

6. **References**: 13 citas BibTeX en `docs/paper/paper.bib`.

## Bibliografía (`docs/paper/paper.bib`)

13 referencias clave en formato BibTeX:

| Clave | Referencia |
|-------|-----------|
| `springel2021simulating` | GADGET-4 (Springel et al. 2021) |
| `teyssier2002cosmological` | RAMSES (Teyssier 2002) |
| `barnes1986hierarchical` | Barnes-Hut tree (Barnes & Hut 1986) |
| `springel2002cosmological` | SPH density-entropy (Springel & Hernquist 2002) |
| `levermore1984relating` | M1 closure (Levermore 1984) |
| `rosdahl2013ramses` | RAMSES-RT (Rosdahl et al. 2013) |
| `harten1983upstream` | HLL scheme (Harten et al. 1983) |
| `anninos1997cosmological` | Química no-equilibrio (Anninos et al. 1997) |
| `verner1996atomic` | Tasas de recombinación (Verner & Ferland 1996) |
| `cen1992hydrodynamic` | Ionización colisional (Cen 1992) |
| `eisenstein1998baryonic` | Función de transferencia (Eisenstein & Hu 1998) |
| `tinker2008toward` | HMF (Tinker et al. 2008) |
| `navarro1997universal` | Perfil NFW (Navarro et al. 1997) |
| `ludlow2016mass` | c-M relation (Ludlow et al. 2016) |
| `lukic2015lyman` | IGM temperatura (Lukić et al. 2015) |

## Próximos pasos para submission JOSS

1. Registrar el código en Zenodo (DOI) y actualizar la referencia.
2. Completar las figuras de validación (P(k), HMF, esfera de Strömgren).
3. Revisar el texto con co-autores y ajustar el Abstract para ≤ 250 palabras.
4. Verificar que el paper compile con Pandoc + LaTeX (formato JOSS estándar).
5. Abrir pre-submission inquiry en JOSS.
