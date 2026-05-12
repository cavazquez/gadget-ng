# Roadmap: Nueva Física e Innovaciones 2024-2025

Este roadmap detalla áreas de vanguardia en física computacional cosmológica y astrofísica, basadas en investigaciones recientes (2024-2025), que pueden integrarse en `gadget-ng` para mantenerlo como un código state-of-the-art.

## 1. Machine Learning y "Simulation-Based Inference" (SBI)

La tendencia más fuerte en 2024-2025 es el uso de IA acoplada a simulaciones N-body e hidrodinámicas (ej. proyecto CAMELS).

*   **Emuladores de Física Submalla (Subgrid):** Reemplazar parametrizaciones costosas (cooling, SFR, feedback) con redes neuronales preentrenadas (Surrogate Models) que predicen las tasas en función de variables locales.
*   **Calibración Automática:** Usar Procesos Gaussianos (Gaussian Processes) para explorar el espacio de parámetros subgrid y calibrarlo automáticamente con observables (ej. función de masa estelar, tamaños galácticos).
*   **Physics-Informed Neural Networks (PINNs):** Usar PINNs para resolver ecuaciones complejas (como transporte radiativo o Schrödinger-Poisson para FDM) en GPUs con mayor velocidad que los solvers de grilla tradicionales.

## 2. Baryon Cycle y Física del Medio Circungaláctico (CGM)

Las observaciones del JWST y eROSITA han revelado que los modelos actuales fallan en predecir el estado termodinámico del CGM.

*   **Química de No-Equilibrio + Difusión Turbulenta:** Modelar explícitamente el no-equilibrio termodinámico de metales acoplado a un modelo de difusión turbulenta en SPH. Esto es crítico para simular la supervivencia de nubes frías en vientos galácticos calientes.
*   **Acoplamiento No Lineal AGN-Feedback Estelar:** Estudios recientes de 2025 muestran que tratar AGN y feedback de SN de forma independiente es insuficiente. Se necesita implementar un acoplamiento donde la turbulencia generada por supernovas modula directamente la tasa de acreción de Bondi del AGN.
*   **CGM Cold Cloudlets (Alta Resolución):** Incorporar técnicas de "cloud-tracking" para resolver estructuras multifásicas muy pequeñas en el CGM que se destruyen con la resolución SPH estándar.

## 3. Dark Matter Más Allá de $\Lambda$CDM

El JWST ha encontrado galaxias masivas muy tempranas y el DESI (2024) ha ajustado restricciones de energía oscura y gravedad.

*   **Fuzzy Dark Matter (FDM) Avanzado:** `gadget-ng` ya tiene proxy FDM, pero la frontera (2025) requiere solvers híbridos (fluido-onda) para resolver la dinámica del "soliton core" vs las interferencias ("granules"). Incorporar integradores que resuelvan el sistema de Schrödinger-Poisson explícitamente para halos tipo Vía Láctea.
*   **Primordial Black Holes (PBHs) como Semillas (Seeds):** Introducir una población inicial de PBHs masivos (~$1000 M_\odot$) en las condiciones iniciales (ICs) que actúen como semillas para los supermasivos (SMBHs) a $z>10$, resolviendo el misterio del JWST de SMBHs "overmassive" tempranos.
*   **Energía Oscura Dinámica Avanzada:** Extender la implementación actual de CPL para incorporar gravedad modificada dependiente de la escala alineada con las restricciones de "Full-Shape" BAO del DESI 2024/2025.

## 4. Evolución del Multiphase ISM (Inspiración COLIBRE)

La nueva suite de simulaciones COLIBRE (2025) ha marcado un antes y un después al eliminar los "temperature floors" artificiales.

*   **Gas Verdaderamente Frío:** Permitir que el ISM se enfríe a $\ll 10^4$ K explícitamente sin presiones efectivas artificiales (como el modelo de Springel & Hernquist), resolviendo el gas molecular directamente.
*   **Modelo de Polvo Activo Completo:** Diferenciar especies de polvo (silicatos de Mg/Fe vs grafitos) y tamaños de grano. Acoplar este polvo *dinámicamente* a la formación de H2 y a la termodinámica del gas (shielding dinámico).

## 5. MHD No Ideal

Las simulaciones de formación estelar e ISM a alta resolución requieren ir más allá de la MHD Ideal implementada en la Phase 123.

*   **Ambipolar Diffusion & Hall Effect:** Introducir difusividad magnética dependiente del estado de ionización del gas. Esto es vital para resolver el problema del "magnetic braking" catastrófico en la formación de discos galácticos y nubes moleculares.

## 6. Métodos Numéricos de Próxima Generación

*   **Meshless Finite Mass (MFM):** Implementar formulaciones "Meshless" Lagrangianas (similares a GIZMO) que combinan las ventajas de SPH (conservación de masa/momento, sin grilla) con la precisión de Riemann solvers (captura de choques precisa sin viscosidad artificial).

---

## Propuesta de Fases (Futuras)

| Fase Propuesta | Descripción | Complejidad | Impacto |
| :--- | :--- | :--- | :--- |
| **Phase 190** | **PBH Seeding para JWST:** Semillas de PBH en ICs y fusión/crecimiento de SMBHs tempranos. | Media | Alto (Papers JWST) |
| **Phase 191** | **Acoplamiento AGN-Stellar Turbulence:** Modificación de `agn.rs` para depender de $v_{disp}$ local. | Baja | Medio |
| **Phase 192** | **Polvo Activo (COLIBRE):** Especies de polvo y acoplamiento termodinámico profundo. | Alta | Alto |
| **Phase 193** | **Emuladores ML (Proof of Concept):** Neural Net en `gadget-ng-physics` para cooling rates rápidos. | Media | Experimental |
| **Phase 194** | **MHD No-Ideal:** Difusión ambipolar dependiente de química. | Muy Alta | Medio |
| **Phase 195** | **MFM (Meshless Finite Mass):** Solver hidrodinámico alternativo a SPH. | Extrema | Transformacional |
