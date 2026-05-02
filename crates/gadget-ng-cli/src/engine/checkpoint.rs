//! Checkpoint save/load for long runs.

use crate::error::CliError;
use gadget_ng_core::Particle;
use gadget_ng_integrators::HierarchicalState;
use gadget_ng_io::{
    JsonlReader, JsonlWriter, Provenance, SnapshotEnv, SnapshotReader, SnapshotWriter,
};
use gadget_ng_parallel::ParallelRuntime;
use std::fs;
use std::path::Path;

// ── Checkpoint ────────────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct CheckpointMeta {
    schema_version: u32,
    /// Último paso completado (el siguiente paso a ejecutar es `completed_step + 1`).
    completed_step: u64,
    /// Factor de escala al final de `completed_step` (1.0 si no hay cosmología).
    a_current: f64,
    /// Hash SHA-256 del TOML canónico, para detectar cambios de config al reanudar.
    config_hash: String,
    /// Número total de partículas (verificación).
    total_particles: usize,
    /// `true` si también se guardó `hierarchical_state.json`.
    has_hierarchical_state: bool,
    /// Informativo: el `SfcDecomposition` no se serializa; se reconstruye al reanudar
    /// desde las posiciones restauradas.  Siempre `false` en el archivo.
    #[serde(default)]
    sfc_state_saved: bool,
    /// `true` si también se guardó `agn_bhs.json` (Phase 106).
    #[serde(default)]
    has_agn_state: bool,
    /// `true` si también se guardó `chem_states.json` (Phase 106).
    #[serde(default)]
    has_chem_state: bool,
}

/// Guarda estado de checkpoint en `<out_dir>/checkpoint/`.
///
/// Solo rank 0 escribe; el directorio se sobreescribe en cada checkpoint
/// (siempre representa el último paso completado).
///
/// Phase 106: incluye estado AGN (`agn_bhs.json`) y química (`chem_states.json`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_checkpoint<R: ParallelRuntime + ?Sized>(
    rt: &R,
    completed_step: u64,
    a_current: f64,
    local: &[Particle],
    total: usize,
    h_state: Option<&HierarchicalState>,
    out_dir: &Path,
    cfg_hash: &str,
    agn_bhs: &[gadget_ng_sph::BlackHole],
    chem_states: &[gadget_ng_rt::ChemState],
) -> Result<(), CliError> {
    let ck_dir = out_dir.join("checkpoint");
    // Recopilar todas las partículas en rank 0 y escribir.
    if let Some(all) = rt.root_gather_particles(local, total) {
        fs::create_dir_all(&ck_dir).map_err(|e| CliError::io(&ck_dir, e))?;
        // Partículas en JSONL (siempre, independientemente del formato de snapshot).
        let dummy_prov = Provenance::new("checkpoint", None, "release", vec![], vec![], cfg_hash);
        let env = SnapshotEnv::default();
        JsonlWriter.write(&ck_dir, &all, &dummy_prov, &env)?;
        // Guardar estado jerárquico si existe.
        if let Some(hs) = h_state {
            hs.save(&ck_dir).map_err(|e| CliError::io(&ck_dir, e))?;
        }
        // Phase 106: guardar estado AGN si hay agujeros negros activos.
        let has_agn = !agn_bhs.is_empty();
        if has_agn {
            let agn_path = ck_dir.join("agn_bhs.json");
            fs::write(&agn_path, serde_json::to_string_pretty(agn_bhs)?)
                .map_err(|e| CliError::io(&agn_path, e))?;
        }
        // Phase 106: guardar estados de química si están activos.
        let has_chem = !chem_states.is_empty();
        if has_chem {
            let chem_path = ck_dir.join("chem_states.json");
            fs::write(&chem_path, serde_json::to_string_pretty(chem_states)?)
                .map_err(|e| CliError::io(&chem_path, e))?;
        }
        // meta.json del checkpoint (diferente al meta.json del snapshot).
        let meta = CheckpointMeta {
            schema_version: 1,
            completed_step,
            a_current,
            config_hash: cfg_hash.to_owned(),
            total_particles: total,
            has_hierarchical_state: h_state.is_some(),
            sfc_state_saved: false,
            has_agn_state: has_agn,
            has_chem_state: has_chem,
        };
        let meta_path = ck_dir.join("checkpoint.json");
        fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
            .map_err(|e| CliError::io(&meta_path, e))?;
    }
    rt.barrier();
    Ok(())
}

/// Carga el estado de checkpoint desde `<resume_dir>/checkpoint/`.
///
/// Devuelve `(partículas_locales, completed_step, a_current, h_state_opt,
///           agn_bhs_opt, chem_states_opt)`.
///
/// Phase 106: incluye estado AGN y química si fueron guardados.
#[allow(clippy::type_complexity)]
pub(crate) fn load_checkpoint<R: ParallelRuntime + ?Sized>(
    rt: &R,
    resume_dir: &Path,
    lo: usize,
    hi: usize,
    cfg_hash: &str,
) -> Result<
    (
        Vec<Particle>,
        u64,
        f64,
        Option<HierarchicalState>,
        Option<Vec<gadget_ng_sph::BlackHole>>,
        Option<Vec<gadget_ng_rt::ChemState>>,
    ),
    CliError,
> {
    let ck_dir = resume_dir.join("checkpoint");
    let meta_path = ck_dir.join("checkpoint.json");
    let meta_str = fs::read_to_string(&meta_path).map_err(|e| CliError::io(&meta_path, e))?;
    let meta: CheckpointMeta = serde_json::from_str(&meta_str)?;
    if meta.config_hash != cfg_hash {
        rt.root_eprintln(&format!(
            "[gadget-ng] ADVERTENCIA: el hash del config ha cambiado \
             desde que se guardó el checkpoint (esperado {}, actual {}). \
             Los resultados pueden diferir.",
            meta.config_hash, cfg_hash
        ));
    }
    // Leer todas las partículas y filtrar las que corresponden a este rango.
    let data = JsonlReader.read(&ck_dir)?;
    let local: Vec<Particle> = data
        .particles
        .into_iter()
        .filter(|p| p.global_id >= lo && p.global_id < hi)
        .collect();
    // Estado jerárquico (opcional).
    let h_state = if meta.has_hierarchical_state {
        Some(HierarchicalState::load(&ck_dir).map_err(|e| CliError::io(&ck_dir, e))?)
    } else {
        None
    };
    // Phase 106: cargar estado AGN si fue guardado.
    let agn_bhs = if meta.has_agn_state {
        let agn_path = ck_dir.join("agn_bhs.json");
        let s = fs::read_to_string(&agn_path).map_err(|e| CliError::io(&agn_path, e))?;
        let bhs: Vec<gadget_ng_sph::BlackHole> = serde_json::from_str(&s)?;
        Some(bhs)
    } else {
        None
    };
    // Phase 106: cargar estados de química si fueron guardados.
    let chem_states = if meta.has_chem_state {
        let chem_path = ck_dir.join("chem_states.json");
        let s = fs::read_to_string(&chem_path).map_err(|e| CliError::io(&chem_path, e))?;
        let cs: Vec<gadget_ng_rt::ChemState> = serde_json::from_str(&s)?;
        Some(cs)
    } else {
        None
    };
    Ok((
        local,
        meta.completed_step,
        meta.a_current,
        h_state,
        agn_bhs,
        chem_states,
    ))
}
