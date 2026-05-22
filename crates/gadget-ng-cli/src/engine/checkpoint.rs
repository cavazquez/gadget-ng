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
#[expect(clippy::too_many_arguments)]
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
#[expect(clippy::type_complexity)]
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

#[cfg(test)]
mod tests {
    use super::CheckpointMeta;

    #[test]
    fn checkpoint_meta_roundtrip_json() {
        let meta = CheckpointMeta {
            schema_version: 1,
            completed_step: 42,
            a_current: 0.5,
            config_hash: "abc123".into(),
            total_particles: 1024,
            has_hierarchical_state: true,
            sfc_state_saved: false,
            has_agn_state: false,
            has_chem_state: true,
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: CheckpointMeta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.completed_step, 42);
        assert!((back.a_current - 0.5).abs() < 1e-12);
        assert!(back.has_hierarchical_state);
        assert!(back.has_chem_state);
    }

    #[test]
    fn save_load_checkpoint_roundtrip_serial() {
        use super::{load_checkpoint, save_checkpoint};
        use crate::config_load;
        use gadget_ng_core::{Particle, RunConfig, Vec3};
        use gadget_ng_parallel::SerialRuntime;

        let cfg: RunConfig = toml::from_str(
            r#"
[simulation]
dt = 0.01
num_steps = 2
softening = 0.05
particle_count = 8
box_size = 1.0
seed = 99

[initial_conditions]
kind = "lattice"
"#,
        )
        .expect("toml parse");
        let hash = config_load::config_canonical_hash(&cfg).expect("hash");
        let rt = SerialRuntime;
        let total = cfg.simulation.particle_count;
        let local: Vec<Particle> = (0..total)
            .map(|i| {
                Particle::new(
                    i,
                    1.0,
                    Vec3::new((i % 2) as f64 * 0.5, ((i / 2) % 2) as f64 * 0.5, 0.5),
                    Vec3::new(0.01, 0.0, 0.0),
                )
            })
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        save_checkpoint(
            &rt,
            2,
            1.0,
            &local,
            total,
            None,
            dir.path(),
            &hash,
            &[],
            &[],
        )
        .expect("save_checkpoint");
        assert!(
            dir.path()
                .join("checkpoint")
                .join("checkpoint.json")
                .exists()
        );

        let (loaded, step, a, h_state, agn, chem) =
            load_checkpoint(&rt, dir.path(), 0, total, &hash).expect("load_checkpoint");
        assert_eq!(loaded.len(), total);
        assert_eq!(step, 2);
        assert!((a - 1.0).abs() < 1e-12);
        assert!(h_state.is_none());
        assert!(agn.is_none());
        assert!(chem.is_none());
        assert_eq!(loaded[0].global_id, 0);
        assert!((loaded[0].mass - 1.0).abs() < 1e-12);
    }

    #[test]
    fn save_load_checkpoint_with_hierarchical_state() {
        use super::{load_checkpoint, save_checkpoint};
        use crate::config_load;
        use gadget_ng_core::{Particle, RunConfig, Vec3};
        use gadget_ng_integrators::HierarchicalState;
        use gadget_ng_parallel::SerialRuntime;

        let cfg: RunConfig = toml::from_str(
            r#"
[simulation]
dt = 0.01
num_steps = 4
softening = 0.05
particle_count = 8
box_size = 1.0
seed = 77

[initial_conditions]
kind = "lattice"

[timestep]
hierarchical = true
"#,
        )
        .expect("toml parse");
        let hash = config_load::config_canonical_hash(&cfg).expect("hash");
        let rt = SerialRuntime;
        let total = cfg.simulation.particle_count;
        let local: Vec<Particle> = (0..total)
            .map(|i| {
                Particle::new(
                    i,
                    1.0,
                    Vec3::new((i % 2) as f64 * 0.5, ((i / 2) % 2) as f64 * 0.5, 0.5),
                    Vec3::zero(),
                )
            })
            .collect();
        let mut h_state = HierarchicalState::new(total);
        h_state.levels[0] = 1;
        h_state.levels[3] = 2;
        h_state.elapsed[1] = 4;
        let dir = tempfile::tempdir().expect("tempdir");
        save_checkpoint(
            &rt,
            3,
            0.25,
            &local,
            total,
            Some(&h_state),
            dir.path(),
            &hash,
            &[],
            &[],
        )
        .expect("save_checkpoint");
        assert!(
            dir.path()
                .join("checkpoint")
                .join("hierarchical_state.json")
                .exists()
        );

        let (loaded, step, a, loaded_h, agn, chem) =
            load_checkpoint(&rt, dir.path(), 0, total, &hash).expect("load_checkpoint");
        assert_eq!(loaded.len(), total);
        assert_eq!(step, 3);
        assert!((a - 0.25).abs() < 1e-12);
        let hs = loaded_h.expect("hierarchical state");
        assert_eq!(hs.levels, h_state.levels);
        assert_eq!(hs.elapsed, h_state.elapsed);
        assert!(agn.is_none());
        assert!(chem.is_none());
    }

    #[test]
    fn save_load_checkpoint_with_agn_and_chem() {
        use super::{load_checkpoint, save_checkpoint};
        use crate::config_load;
        use gadget_ng_core::{Particle, RunConfig, Vec3};
        use gadget_ng_parallel::SerialRuntime;
        use gadget_ng_rt::ChemState;
        use gadget_ng_sph::BlackHole;

        let cfg: RunConfig = toml::from_str(
            r#"
[simulation]
dt = 0.01
num_steps = 2
softening = 0.05
particle_count = 4
box_size = 1.0
seed = 88

[initial_conditions]
kind = "lattice"
"#,
        )
        .expect("toml parse");
        let hash = config_load::config_canonical_hash(&cfg).expect("hash");
        let rt = SerialRuntime;
        let total = cfg.simulation.particle_count;
        let local: Vec<Particle> = (0..total)
            .map(|i| Particle::new(i, 1.0, Vec3::new(i as f64 * 0.2, 0.0, 0.0), Vec3::zero()))
            .collect();
        let agn_bhs = vec![BlackHole::with_spin(Vec3::new(0.5, 0.5, 0.5), 1.0e6, 0.3)];
        let mut chem_states = vec![ChemState::neutral(); total];
        chem_states[1] = ChemState::fully_ionized();
        let dir = tempfile::tempdir().expect("tempdir");
        save_checkpoint(
            &rt,
            1,
            1.0,
            &local,
            total,
            None,
            dir.path(),
            &hash,
            &agn_bhs,
            &chem_states,
        )
        .expect("save_checkpoint");
        let ck = dir.path().join("checkpoint");
        assert!(ck.join("agn_bhs.json").exists());
        assert!(ck.join("chem_states.json").exists());

        let (loaded, step, _, h_state, loaded_agn, loaded_chem) =
            load_checkpoint(&rt, dir.path(), 0, total, &hash).expect("load_checkpoint");
        assert_eq!(loaded.len(), total);
        assert_eq!(step, 1);
        assert!(h_state.is_none());
        let bhs = loaded_agn.expect("agn bhs");
        assert_eq!(bhs.len(), 1);
        assert!((bhs[0].mass - 1.0e6).abs() < 1.0);
        assert!((bhs[0].spin - 0.3).abs() < 1e-12);
        let chem = loaded_chem.expect("chem states");
        assert_eq!(chem.len(), total);
        assert!(chem[1].x_hii > 0.9);
    }
}
