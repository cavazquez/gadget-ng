# Primeros pasos con gadget-ng

Guía mínima para tener tu primera simulación corriendo en menos de 10 minutos.

---

## 1. Requisitos previos

| Herramienta | Versión mínima | Instalación |
|-------------|---------------|-------------|
| Rust        | 1.74+         | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Python      | 3.10+         | Para postproceso y visualización (opcional) |
| libhdf5-dev | 1.10+         | Solo si quieres snapshots HDF5 (opcional) |
| OpenMPI     | 4.0+          | Solo para simulaciones distribuidas (opcional) |

---

## 2. Compilar el binario

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/gadget-ng.git
cd gadget-ng

# Compilar en modo release (recomendado, ~2-3 min la primera vez)
cargo build --release -p gadget-ng-cli

# El binario queda en:
./target/release/gadget-ng --help
```

---

## 3. Tu primera simulación: esfera de Plummer

La esfera de Plummer es el ejemplo más simple: un sistema auto-gravitante en
equilibrio virial. 512 partículas, sin cosmología.

```bash
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer \
  --snapshot
```

Deberías ver en pantalla el progreso de los 200 pasos. Al terminar, en
`runs/plummer/` encontrarás:

- `snapshot_final/` — estado final de las partículas
- `checkpoint.json` — metadatos del run (paso, semilla, hash de config)

**Visualizar el resultado:**

```bash
./target/release/gadget-ng visualize \
  --snapshot runs/plummer/snapshot_final \
  --output runs/plummer/frame.png \
  --color velocity
```

---

## 4. Órbita kepleriana (2 cuerpos)

Un sistema de 2 partículas con gravedad directa. Verifica que el período
orbital y la energía se conservan.

```bash
./target/release/gadget-ng stepping \
  --config examples/kepler_orbit.toml \
  --out runs/kepler \
  --snapshot
```

---

## 5. N-body con análisis (1000 partículas)

Incluye cálculo in-situ de espectro de potencia P(k) y catálogo FoF de halos.

```bash
# Simulación
./target/release/gadget-ng stepping \
  --config examples/nbody_bh_1k.toml \
  --out runs/nbody \
  --snapshot

# Análisis completo: FoF + P(k) + función de correlación ξ(r) + relación c(M)
./target/release/gadget-ng analyze \
  --snapshot runs/nbody/snapshot_final \
  --out runs/nbody/analysis
```

Resultado en `runs/nbody/analysis/results.json`.

---

## 6. Simulación cosmológica ΛCDM

512 partículas de materia oscura, caja periódica de 100 Mpc/h,
desde z=49 hasta z=0.

```bash
./target/release/gadget-ng stepping \
  --config examples/cosmological.toml \
  --out runs/cosmo \
  --snapshot
```

**Postprocesar el espectro de potencia:**

```bash
# Instalar dependencias Python (ver requirements.txt)
pip install -r requirements.txt

python docs/scripts/postprocess_pk.py \
  --insitu runs/cosmo/insitu \
  --out    runs/cosmo/analysis/pk_evolution.json
```

---

## 7. Reanudar desde un checkpoint

Todos los runs generan checkpoints automáticamente. Para reanudar:

```bash
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer \
  --resume runs/plummer
```

---

## 8. Validar la instalación con los tests

```bash
# Tests rápidos del core
cargo test -p gadget-ng-core

# Suite completa de validación física (~3.5 min)
cargo test -p gadget-ng-physics
```

---

## Próximos pasos

| Quiero... | Lee... |
|-----------|--------|
| Entender todas las opciones del TOML | [README.md — Configuración TOML](../README.md#configuración-toml) |
| Usar MPI para simulaciones grandes | [runbooks/mpi-cluster.md](runbooks/mpi-cluster.md) |
| Explorar la física implementada | [physics-roadmap.md](physics-roadmap.md) |
| Correr ejemplos con notebooks interactivos | [notebooks/](../notebooks/) |
| Ver la arquitectura del código | [architecture.md](architecture.md) |

---

> **Nota sobre unidades:** Cuando `[units]` está activado en el TOML, el
> sistema trabaja en kpc / M☉ / km·s⁻¹ con G calculado de forma coherente.
> Sin `[units]`, todas las cantidades son adimensionales (N-body units).
