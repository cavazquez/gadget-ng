#!/usr/bin/env bash
# ── Fase 23: TreePM SR-SFC — Script de ejecución y comparación ────────────────
#
# Ejecuta tres paths para comparación:
#   A) Fase 21: treepm_slab, halo 1D-z
#   B) Fase 22: treepm_slab, halo 3D periódico
#   C) Fase 23: treepm_sr_sfc, halo 3D periódico, dominio SFC
#
# Produce tablas comparativas de:
#   - wall time (s/paso)
#   - halo particles y bytes
#   - sr_sync_ns (costo sincronización PM↔SR, solo Fase 23)
#   - v_rms, delta_rms (física cosmológica)
#
# Uso:
#   bash run_phase23.sh              # P=1 (serial)
#   NPROC=2 bash run_phase23.sh      # P=2 (MPI, requiere --features mpi)
#   NPROC=4 bash run_phase23.sh      # P=4 (MPI)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CONFIGS_DIR="$SCRIPT_DIR/configs"

NPROC="${NPROC:-1}"
BINARY="${BINARY:-$REPO_ROOT/target/release/gadget-ng}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Fase 23 — TreePM SR-SFC: Benchmark comparativo"
echo "  NPROC=$NPROC  | $(date)"
echo "═══════════════════════════════════════════════════════════════"

# Compilar en modo release.
echo "[1/5] Compilando gadget-ng (release)..."
cd "$REPO_ROOT"
cargo build --release -p gadget-ng-cli 2>&1 | tail -3

mkdir -p "$RESULTS_DIR"

run_sim() {
    local label="$1"
    local config="$2"
    local outdir="$RESULTS_DIR/$label"
    mkdir -p "$outdir"

    echo ""
    echo "─── Ejecutando: $label (P=$NPROC) ───"
    if [ "$NPROC" -gt 1 ]; then
        mpirun -n "$NPROC" "$BINARY" run "$config" "$outdir"
    else
        "$BINARY" run "$config" "$outdir"
    fi

    # Extraer métricas clave de timings.json
    if [ -f "$outdir/timings.json" ]; then
        echo "  Métricas ($label):"
        python3 -c "
import json, sys
with open('$outdir/timings.json') as f:
    t = json.load(f)
hpc = t.get('hpc', {})
print(f'    mean_step_wall_s  = {t.get(\"mean_step_wall_s\", 0):.4f} s')
print(f'    path_active       = {hpc.get(\"path_active\", \"?\")}')
print(f'    mean_short_range_halo_particles = {hpc.get(\"mean_short_range_halo_particles\", 0):.1f}')
print(f'    mean_short_range_halo_bytes     = {hpc.get(\"mean_short_range_halo_bytes\", 0):.0f} B')
print(f'    mean_tree_short_s = {hpc.get(\"mean_tree_short_s\", 0):.4f} s')
print(f'    mean_pm_long_s    = {hpc.get(\"mean_pm_long_s\", 0):.4f} s')
print(f'    mean_sr_sync_s    = {hpc.get(\"mean_sr_sync_s\", 0):.4f} s')
print(f'    mean_sr_domain_particle_count = {hpc.get(\"mean_sr_domain_particle_count\", 0):.1f}')
" 2>/dev/null || echo "  (timings.json no disponible)"
    fi

    # Extraer última línea de diagnósticos para v_rms y delta_rms.
    if [ -f "$outdir/diagnostics.jsonl" ]; then
        python3 -c "
import json
lines = open('$outdir/diagnostics.jsonl').readlines()
if lines:
    d = json.loads(lines[-1])
    print(f'    v_rms     = {d.get(\"v_rms\", 0):.4e}')
    print(f'    delta_rms = {d.get(\"delta_rms\", 0):.4e}')
    print(f'    a         = {d.get(\"a\", 0):.4f}')
" 2>/dev/null || echo "  (diagnostics.jsonl no disponible)"
    fi
}

# ── A) Baseline Fase 21: slab 1D-z ──────────────────────────────────────────
run_sim "fase21_slab_1d" "$CONFIGS_DIR/eds_N512_slab_1d_baseline.toml"

# ── B) Baseline Fase 22: slab + halo 3D ─────────────────────────────────────
run_sim "fase22_slab_3d" "$CONFIGS_DIR/eds_N512_slab_3d_baseline.toml"

# ── C) Fase 23: SR-SFC + halo 3D ────────────────────────────────────────────
run_sim "fase23_sr_sfc_p1" "$CONFIGS_DIR/eds_N512_sr_sfc_p1.toml"

# ── Tabla comparativa ────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Tabla comparativa (N=512, EdS)"
echo "═══════════════════════════════════════════════════════════════"
python3 "$SCRIPT_DIR/scripts/compare_results.py" \
    "$RESULTS_DIR/fase21_slab_1d" \
    "$RESULTS_DIR/fase22_slab_3d" \
    "$RESULTS_DIR/fase23_sr_sfc_p1" \
    2>/dev/null || echo "(análisis Python no disponible)"

echo ""
echo "Resultados en: $RESULTS_DIR"
echo "Fase 23 completada: $(date)"
