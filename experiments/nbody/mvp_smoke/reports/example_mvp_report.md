# Informe de ejemplo — MVP `gadget-ng`

**Fecha**: 2026-04-15 (ejemplo autogenerado en el repositorio)

## Resumen

Se ejecutó la cadena `scripts/check.sh` y la validación `compare_serial_mpi.sh` en un entorno de desarrollo con OpenMPI. El binario `gadget-ng` reproduce resultados idénticos (en `f64`) entre ejecución serial y MPI (4 rangos) para el experimento `parity.toml`.

## Comandos ejecutados (referencia)

```bash
./scripts/check.sh
./scripts/validation/compare_serial_mpi.sh
```

## Resultado

- `max_abs_diff` (posiciones, velocidades, masas en snapshot): **0.0** frente a umbral `1e-12`.

## Próximos pasos sugeridos

- Aumentar `particle_count` en pasos controlados y registrar tiempo de pared.
- Añadir solver no \(O(N^2)\) cuando la capa `GravitySolver` tenga segunda implementación.
