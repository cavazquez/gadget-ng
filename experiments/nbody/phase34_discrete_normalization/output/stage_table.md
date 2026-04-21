| # | Etapa | Observable | Esperado | Medido | Extra | Comentario |
|---|-------|------------|----------|--------|-------|------------|
| 1 | continuo → δ̂(k) discreto | ⟨|δ̂|²⟩ / (σ²·N³) | 1.0000 | 0.9959 | — | Test ruido blanco, N=32, σ=0.1. Confirma `forward unnormalized ⇒ Var(δ̂) = σ²·N³`. |
| 2 | δ̂(k) → δ(x) → FFT | max |δ_out - δ_in| | 0.000e+00 | 8.882e-16 | — | Roundtrip a precisión máquina (N=16); modo único recuperado con error 9.16e-16. |
| 3 | grilla pura: P_grid / P_cont | A_grid = ⟨P_grid / P_cont⟩ | 2·V²/N⁹ (convención del código) | 5.508e-14 | CV/raw: 0.0235 | N=32, 6 seeds. El factor 2 viene de σ²→⟨|δ̂|²⟩=2σ². CV entre seeds: 0.023. |
| 4 | grilla → partículas ZA + CIC + deconv | A_part / A_grid | ≈ 1 si partículas reproducen δ̂ fielmente | 0.0301 | CV/raw: 0.0061 | Factor multiplicativo exclusivo del paso a partículas: 0.0301 (CV 0.0061). Extremadamente determinista: 6 seeds. |
| 5 | deconvolución CIC | pendiente de log R(k) vs log k | 0.000e+00 | -0.0942 | CV/raw: -0.1855 | slope_raw=-0.1855 → slope_deconv=-0.0942 (reducción 49.2 %). El CIC introduce la única dependencia en k residual. |
| 6 | offset global aislado (sin solver) | CV(P_m/P_cont) en k ≤ k_Nyq/2 | < 0.15 | 0.1104 | — | A_mean=1.985e-15, 8 bins. Confirma que la distorsión de forma es << 1 respecto al offset global. |
| 7 | escalado con resolución N | log₁₀(A₁₆/A₃₂) | 2.7093 | 3.2964 | — | N∈{16,32}, 6 seeds. Observado 3.296 vs predicción 2.709; exceso ≈ 0.587 décadas. |
| 8 | determinismo entre seeds | CV(A) sobre 6 seeds | < 0.10 | 0.0519 | — | CV=0.0519 → A es determinista, no estadístico. |
