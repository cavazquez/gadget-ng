# Phase 173 — Auditoría CLI `--use-nonlinear` (2026-05-12)

Comportamiento con `gadget-ng fisher` antes de `--pk-model`:

| Invocación | `config.use_nonlinear` en JSON de salida |
|------------|------------------------------------------|
| Sin `--use-nonlinear` | `true` (Halofit por defecto) |
| Con `--use-nonlinear` | `true` |
| `--use-nonlinear false` | **Error**: Clap no acepta argumento adicional; no había forma de elegir P(k) lineal |

Conclusión: el flag `bool` con `default_value_t = true` no permitía desactivar Halofit desde la CLI; se reemplaza por `--pk-model linear|nonlinear`.
