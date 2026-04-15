//! Layout SoA (Structure of Arrays) para transferencia eficiente host ↔ device.
//!
//! Este módulo no depende de `gadget-ng-core`. Las conversiones hacia/desde
//! `Particle` viven en `gadget_ng_core::gpu_bridge` (dentro del crate core,
//! bajo `feature = "gpu"`), lo que evita la dependencia circular.

/// Representación SoA de un conjunto de partículas N-body.
///
/// Todos los arrays tienen la misma longitud `n = ids.len()`.
/// El índice `i` corresponde a la misma partícula en todos los arrays.
///
/// Esta estructura está diseñada para transferencias DMA eficientes host → device:
/// cada array es un bloque contiguo de `f64` listo para mapear a un buffer GPU.
#[derive(Debug, Clone, Default)]
pub struct GpuParticlesSoA {
    /// Posición X de cada partícula.
    pub xs: Vec<f64>,
    /// Posición Y de cada partícula.
    pub ys: Vec<f64>,
    /// Posición Z de cada partícula.
    pub zs: Vec<f64>,
    /// Velocidad X de cada partícula.
    pub vxs: Vec<f64>,
    /// Velocidad Y de cada partícula.
    pub vys: Vec<f64>,
    /// Velocidad Z de cada partícula.
    pub vzs: Vec<f64>,
    /// Masa de cada partícula.
    pub masses: Vec<f64>,
    /// `global_id` de cada partícula (índice en [0, N)).
    pub ids: Vec<usize>,
}

impl GpuParticlesSoA {
    /// Construye un `GpuParticlesSoA` directamente desde arrays planos.
    ///
    /// Todos los slices deben tener la misma longitud; de lo contrario este
    /// constructor entra en pánico. La conversión desde `&[Particle]` está en
    /// `gadget_ng_core::GpuParticlesSoA::from_particles` (cuando la feature `gpu`
    /// está activa en `gadget-ng-core`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(
        xs: Vec<f64>,
        ys: Vec<f64>,
        zs: Vec<f64>,
        vxs: Vec<f64>,
        vys: Vec<f64>,
        vzs: Vec<f64>,
        masses: Vec<f64>,
        ids: Vec<usize>,
    ) -> Self {
        let n = ids.len();
        assert_eq!(xs.len(), n);
        assert_eq!(ys.len(), n);
        assert_eq!(zs.len(), n);
        assert_eq!(vxs.len(), n);
        assert_eq!(vys.len(), n);
        assert_eq!(vzs.len(), n);
        assert_eq!(masses.len(), n);
        Self {
            xs,
            ys,
            zs,
            vxs,
            vys,
            vzs,
            masses,
            ids,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_arrays_ok() {
        let soa = GpuParticlesSoA::from_arrays(
            vec![0.1, 4.0],
            vec![0.2, 5.0],
            vec![0.3, 6.0],
            vec![1.0, -1.0],
            vec![2.0, 0.0],
            vec![3.0, 1.0],
            vec![1.0, 2.0],
            vec![0, 1],
        );
        assert_eq!(soa.len(), 2);
        assert!(!soa.is_empty());
        assert!((soa.xs[0] - 0.1).abs() < 1e-15);
        assert!((soa.ys[1] - 5.0).abs() < 1e-15);
        assert_eq!(soa.ids[1], 1);
    }

    #[test]
    fn empty_soa() {
        let soa = GpuParticlesSoA::default();
        assert!(soa.is_empty());
        assert_eq!(soa.len(), 0);
    }
}
