# Validación ligera: P(k) y \(\sigma_8\) frente a referencia tipo GADGET-4

Runbook para reproducir localmente la cadena **corrida acotada → `insitu_*.json` → métricas** usando el script ya incluido en el repo. Las comparaciones son **referencia**, no igualdad con otro binario.

## Requisitos

- Binario `gadget-ng` compilado en release (`cargo build --release -p gadget-ng-cli`).
- Python 3 con dependencias mínimas para el script (véase cabecera de `docs/scripts/bench_pk_vs_gadget4.py`).
- Opcional: MPI (`mpirun`) si quieres repetir la validación en paralelo.

## 1. Correr una validación acotada

El repo incluye una corrida tipo \(\Lambda\)CDM \(N=128^3\) y análisis in-situ:

```bash
./scripts/run_validation_128.sh
```

Por defecto usa `configs/validation_128.toml`, escribe bajo `runs/validation_128/` y, si existe `[insitu_analysis].output_dir` en el TOML, los ficheros **`insitu_*.json`** van a ese directorio (en la configuración estándar: `runs/validation_128/analysis/`).

Para una corrida más corta orientada a CI o máquinas modestas, existe `configs/validation_128_test.toml`; puedes copiar el script y apuntar `CONFIG=` a ese archivo o invocar `gadget-ng stepping` manualmente:

```bash
./target/release/gadget-ng stepping \
  --config configs/validation_128.toml \
  --out runs/validation_128
```

Asegúrate de que en el TOML figure **`[insitu_analysis] enabled = true`** y un **`output_dir`** conocido (o omítelo y usa por defecto `<out_dir>/insitu/`).

## 2. Ejecutar el benchmark P(k) vs GADGET-4

```bash
python3 docs/scripts/bench_pk_vs_gadget4.py \
  --insitu-dir runs/validation_128/analysis \
  --output bench_results/pk_comparison.json
```

Ajusta `--insitu-dir` al directorio donde están los `insitu_*.json` del paso anterior.

Opcional: generar una figura PNG si el script lo permite:

```bash
python3 docs/scripts/bench_pk_vs_gadget4.py \
  --insitu-dir runs/validation_128/analysis \
  --output bench_results/pk_comparison.json \
  --plot bench_results/pk_comparison.png
```

## 3. Interpretar la salida

El JSON incluye (entre otros):

- **`metrics_z0`**: error relativo de \(\sigma_8\) respecto a la tabla de referencia embebida en el script, estadísticas del ratio \(P_\mathrm{sim}(k)/P_\mathrm{ref}(k)\).
- **`sigma8_evolution`**: valores inferidos por snapshot si hay P(k) en cada uno.

Los umbrales “buenos” dependen del caso físico y de la resolución; el propio script imprime un resumen en consola. Úsalo como comprobación rápida antes de corridas mayores.

## 4. Contexto del equipo

Para resultados históricos de benchmarks MPI y discusión más amplia, ver el informe [docs/reports/2026-04-phase92-benchmarks-rsmpi.md](../reports/2026-04-phase92-benchmarks-rsmpi.md) si sigue vigente en tu línea de trabajo.

## 5. Post-proceso alternativo del mismo run

El script `scripts/run_validation_128.sh` admite `--post` para lanzar `docs/scripts/validate_pk_hmf.py` y `postprocess_pk.py` cuando están presentes; es complementario al benchmark anterior (CLASS/HMF frente a otros objetivos).
