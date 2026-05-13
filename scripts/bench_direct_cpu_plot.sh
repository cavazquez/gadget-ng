#!/usr/bin/env bash
# Wrapper: gravedad directa — benchmarks Criterion (--quick) + CSV + gráfico comparativo.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$SCRIPT_DIR/bench_direct_cpu_plot.py"
