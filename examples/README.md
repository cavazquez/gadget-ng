# Ejemplos

Configuraciones TOML listas para usar con el binario `gadget-ng`.

| Archivo | Partículas | Solver | Descripción |
|---------|-----------|--------|-------------|
| [`plummer_sphere.toml`](plummer_sphere.toml) | 512 | Barnes-Hut θ=0.5 | Esfera de Plummer en equilibrio virial |
| [`kepler_orbit.toml`](kepler_orbit.toml) | 2 | Directo | Órbita circular Sol-Tierra |
| [`nbody_bh_1k.toml`](nbody_bh_1k.toml) | 1000 | Barnes-Hut θ=0.4 | N-body retícula + análisis FoF/P(k) |
| [`cosmological.toml`](cosmological.toml) | 512 | Barnes-Hut | ΛCDM z=49→0 con integración cosmológica |

## Inicio rápido

```bash
# Compilar en modo release (recomendado)
cargo build --release -p gadget-ng-cli

# Esfera de Plummer
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --snapshot

# Visualizar
./target/release/gadget-ng visualize \
  --snapshot runs/plummer/snapshot_final \
  --output runs/plummer/frame.png --color velocity

# Órbita Kepleriana
./target/release/gadget-ng stepping \
  --config examples/kepler_orbit.toml \
  --out runs/kepler --snapshot

# N-body con análisis
./target/release/gadget-ng stepping \
  --config examples/nbody_bh_1k.toml \
  --out runs/nbody --snapshot

./target/release/gadget-ng analyse \
  --snapshot runs/nbody/snapshot_final \
  --out runs/nbody/analysis
```
