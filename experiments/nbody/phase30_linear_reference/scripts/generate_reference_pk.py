#!/usr/bin/env python3
"""
generate_reference_pk.py — Fase 30: Referencia externa EH + CAMB opcional
==========================================================================

Genera el espectro de potencia de referencia para comparar con gadget-ng.

Implementa el espectro Eisenstein-Hu no-wiggle en Python, de forma
independiente al código Rust, y opcionalmente usa CAMB como referencia
externa genuina.

## Uso

    python generate_reference_pk.py [opciones] --out reference_pk.json

## Cosmología de referencia exacta (Planck 2018)

    Omega_m  = 0.315
    Omega_b  = 0.049
    h        = 0.674
    n_s      = 0.965
    sigma8   = 0.8
    T_CMB    = 2.7255 K

## Salida JSON

    {
      "cosmology": {...},
      "z": ...,
      "source": "EH" | "CAMB",
      "k_units": "h/Mpc",
      "pk_units": "(Mpc/h)^3",
      "sigma8_target": 0.8,
      "sigma8_integral": ...,
      "bins": [{"k": ..., "pk": ...}, ...]
    }
"""

import argparse
import json
import math
import sys

# ── Parámetros por defecto ────────────────────────────────────────────────────

DEFAULT_OMEGA_M  = 0.315
DEFAULT_OMEGA_B  = 0.049
DEFAULT_H        = 0.674
DEFAULT_N_S      = 0.965
DEFAULT_SIGMA8   = 0.8
DEFAULT_T_CMB    = 2.7255
DEFAULT_Z        = 0.0          # z=0 por defecto para el espectro de referencia
DEFAULT_K_MIN    = 1e-4         # h/Mpc
DEFAULT_K_MAX    = 10.0         # h/Mpc
DEFAULT_N_K      = 200          # bins logarítmicos


# ── Función de transferencia EH no-wiggle ─────────────────────────────────────

def transfer_eh_nowiggle(k_hmpc, omega_m, omega_b, h):
    """
    Función de transferencia Eisenstein-Hu no-wiggle (1998, ApJ 496, 605).

    Parámetros
    ----------
    k_hmpc : float
        Número de onda en h/Mpc.
    omega_m : float
        Densidad de materia Omega_m (sin dimensiones).
    omega_b : float
        Densidad de bariones Omega_b (sin dimensiones).
    h : float
        Parámetro de Hubble adimensional (H0/100).

    Retorna
    -------
    float
        T(k) ∈ (0, 1].
    """
    if k_hmpc <= 0.0:
        return 1.0

    omega_m_h2 = omega_m * h * h
    omega_b_h2 = omega_b * h * h
    fb = omega_b / omega_m

    # Horizonte de sonido aproximado [Mpc]
    s = (44.5 * math.log(9.83 / omega_m_h2)
         / math.sqrt(1.0 + 10.0 * omega_b_h2 ** 0.75))

    # Supresión bariónica del shape parameter
    alpha_gamma = (1.0
                   - 0.328 * math.log(431.0 * omega_m_h2) * fb
                   + 0.38  * math.log(22.3  * omega_m_h2) * fb * fb)

    # k en Mpc^-1
    k_mpc = k_hmpc * h
    ks = 0.43 * k_mpc * s

    gamma_eff = omega_m * h * (alpha_gamma + (1.0 - alpha_gamma)
                                / (1.0 + ks**4))
    gamma_eff = max(gamma_eff, 1e-30)

    q = k_hmpc / gamma_eff

    l0 = math.log(2.0 * math.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    t  = l0 / (l0 + c0 * q * q)

    return max(0.0, min(1.0, t))


# ── Integral sigma8 ───────────────────────────────────────────────────────────

def tophat_window(x):
    """Filtro top-hat en k-space: W(x) = 3[sin(x) - x cos(x)] / x³."""
    if abs(x) < 1e-4:
        return 1.0 - x * x / 10.0
    return 3.0 * (math.sin(x) - x * math.cos(x)) / (x * x * x)


def sigma_sq(amp, n_s, omega_m, omega_b, h, r_mpc_h=8.0, n_steps=8192):
    """
    Calcula σ²(R) para P(k) = amp² · k^n_s · T²(k).

    La integral se hace en log-k desde k=1e-5 a 500 h/Mpc.

    Retorna sigma²(R) adimensional.
    """
    k_min = 1e-5
    k_max = 5e2
    ln_k_min = math.log(k_min)
    ln_k_max = math.log(k_max)
    d_ln_k = (ln_k_max - ln_k_min) / n_steps

    total = 0.0
    k_prev = k_min
    tk_prev = transfer_eh_nowiggle(k_prev, omega_m, omega_b, h)
    w_prev = tophat_window(k_prev * r_mpc_h)
    f_prev = k_prev**(n_s + 3.0) * tk_prev**2 * w_prev**2

    for i in range(1, n_steps + 1):
        k = math.exp(ln_k_min + i * d_ln_k)
        tk = transfer_eh_nowiggle(k, omega_m, omega_b, h)
        w  = tophat_window(k * r_mpc_h)
        f  = k**(n_s + 3.0) * tk**2 * w**2
        total += 0.5 * (f_prev + f) * d_ln_k
        f_prev = f

    return amp * amp * total / (2.0 * math.pi**2)


def amplitude_for_sigma8(sigma8_target, n_s, omega_m, omega_b, h):
    """
    Calcula la amplitud A tal que σ(8 Mpc/h) = sigma8_target.

    P(k) = A² · k^n_s · T²(k)
    """
    # sigma_sq_unit = σ²(8, A=1) = integral con amp=1
    sq_unit = sigma_sq(1.0, n_s, omega_m, omega_b, h, r_mpc_h=8.0)
    if sq_unit <= 0.0:
        return sigma8_target
    return sigma8_target / math.sqrt(sq_unit)


def sigma_from_bins(bins_k_pk, r_mpc_h=8.0):
    """
    Calcula σ(R) desde bins de P(k).

    bins_k_pk : list of (k [h/Mpc], P(k) [(Mpc/h)³])
    """
    if not bins_k_pk:
        return 0.0
    total = 0.0
    n = len(bins_k_pk)
    for i, (k, pk) in enumerate(bins_k_pk):
        if pk <= 0.0 or k <= 0.0:
            continue
        w = tophat_window(k * r_mpc_h)
        if n > 1:
            if i == 0:
                dk = bins_k_pk[1][0] - bins_k_pk[0][0]
            elif i == n - 1:
                dk = bins_k_pk[n-1][0] - bins_k_pk[n-2][0]
            else:
                dk = 0.5 * (bins_k_pk[i+1][0] - bins_k_pk[i-1][0])
        else:
            dk = k * 0.1
        total += k * k * pk * w * w * dk
    return math.sqrt(total / (2.0 * math.pi**2))


# ── Generación del espectro EH ────────────────────────────────────────────────

def generate_eh_spectrum(omega_m, omega_b, h, n_s, sigma8, z,
                         k_min=DEFAULT_K_MIN, k_max=DEFAULT_K_MAX,
                         n_k=DEFAULT_N_K):
    """
    Genera el espectro lineal EH no-wiggle normalizado a sigma8.

    El factor de crecimiento D(z) se aplica para escalar al redshift z.
    Usa la aproximación D(z) ≈ D(0) × g(z) donde g(z) es la función de
    crecimiento de Carroll, Press & Turner (1992).
    """
    amp = amplitude_for_sigma8(sigma8, n_s, omega_m, omega_b, h)

    # Factor de crecimiento (aproximación CPT92)
    omega_lambda = 1.0 - omega_m  # ΛCDM plano
    def growth_factor_cpt92(z):
        a = 1.0 / (1.0 + z)
        om_a = omega_m / (omega_m + omega_lambda * a**3)
        ol_a = omega_lambda * a**3 / (omega_m + omega_lambda * a**3)
        # Carroll, Press & Turner (1992) Eq. 29
        g = 2.5 * om_a / (
            om_a**(4.0/7.0) - ol_a
            + (1.0 + om_a/2.0) * (1.0 + ol_a/70.0)
        )
        return g

    g0 = growth_factor_cpt92(0.0)
    gz = growth_factor_cpt92(z)
    d_ratio = gz / g0  # D(z)/D(0)

    bins = []
    for i in range(n_k):
        k = k_min * (k_max / k_min) ** (i / (n_k - 1))
        tk = transfer_eh_nowiggle(k, omega_m, omega_b, h)
        pk_z0 = amp * amp * k**n_s * tk**2
        pk_z  = pk_z0 * d_ratio**2
        bins.append({"k": k, "pk": pk_z})

    # Verificar sigma8 a z=0
    sigma8_check = sigma_from_bins(
        [(b["k"], b["pk"] / d_ratio**2) for b in bins]
    )

    return bins, sigma8_check, amp


# ── Intento con CAMB ──────────────────────────────────────────────────────────

def try_generate_camb_spectrum(omega_m, omega_b, h, n_s, sigma8, z,
                               k_min=DEFAULT_K_MIN, k_max=DEFAULT_K_MAX,
                               n_k=DEFAULT_N_K):
    """
    Intenta generar el espectro con CAMB. Retorna (bins, info_dict) o None.

    Requiere: pip install camb
    """
    try:
        import camb  # noqa: F401
    except ImportError:
        return None, "CAMB no disponible (pip install camb)"

    try:
        import camb
        import numpy as np

        # Parámetros CAMB
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=h * 100.0,
            ombh2=omega_b * h**2,
            omch2=(omega_m - omega_b) * h**2,
            mnu=0.0,
            omk=0.0,
            TCMB=2.7255,
        )
        pars.InitPower.set_params(ns=n_s)
        pars.set_matter_power(redshifts=[z], kmax=k_max * 2.0)
        pars.NonLinear = camb.model.NonLinear_none

        results = camb.get_results(pars)

        # Normalizar a sigma8
        sigma8_camb = results.get_sigma8_0()
        k_arr, z_arr, pk_arr = results.get_matter_power_spectrum(
            minkh=k_min, maxkh=k_max, npoints=n_k
        )
        pk_arr_z = pk_arr[0]  # z=z_arr[0]

        # Escalar a sigma8 objetivo
        scale = (sigma8 / sigma8_camb) ** 2
        pk_scaled = pk_arr_z * scale

        bins = [{"k": float(k), "pk": float(pk)}
                for k, pk in zip(k_arr, pk_scaled)]

        sigma8_check = sigma_from_bins(list(zip(k_arr, pk_scaled / scale)))

        info = {
            "camb_version": camb.__version__,
            "sigma8_camb_original": float(sigma8_camb),
            "scale_factor": float(scale),
        }
        return bins, info

    except Exception as e:
        return None, f"Error CAMB: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera espectro de referencia EH (+CAMB opcional)"
    )
    parser.add_argument("--omega-m",  type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--omega-b",  type=float, default=DEFAULT_OMEGA_B)
    parser.add_argument("--h",        type=float, default=DEFAULT_H)
    parser.add_argument("--n-s",      type=float, default=DEFAULT_N_S)
    parser.add_argument("--sigma8",   type=float, default=DEFAULT_SIGMA8)
    parser.add_argument("--z",        type=float, default=DEFAULT_Z,
                        help="Redshift para evaluar el espectro")
    parser.add_argument("--k-min",    type=float, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max",    type=float, default=DEFAULT_K_MAX)
    parser.add_argument("--n-k",      type=int,   default=DEFAULT_N_K)
    parser.add_argument("--out",      type=str,   default="reference_pk.json",
                        help="Archivo JSON de salida")
    parser.add_argument("--try-camb", action="store_true",
                        help="Intentar usar CAMB como referencia alternativa")
    args = parser.parse_args()

    cosmo = {
        "omega_m":  args.omega_m,
        "omega_b":  args.omega_b,
        "h":        args.h,
        "n_s":      args.n_s,
        "sigma8":   args.sigma8,
        "t_cmb":    DEFAULT_T_CMB,
    }

    print(f"[generate_reference_pk] Cosmología: {cosmo}")
    print(f"  z = {args.z},  k ∈ [{args.k_min}, {args.k_max}] h/Mpc,  N_k = {args.n_k}")

    # ── Generar espectro EH ─────────────────────────────────────────────────
    bins_eh, sigma8_check, amp = generate_eh_spectrum(
        args.omega_m, args.omega_b, args.h, args.n_s, args.sigma8, args.z,
        k_min=args.k_min, k_max=args.k_max, n_k=args.n_k,
    )
    print(f"  EH: sigma8_integral (z=0 check) = {sigma8_check:.4f} (target {args.sigma8})")
    print(f"  EH: A = {amp:.4e}")

    result = {
        "cosmology":      cosmo,
        "z":              args.z,
        "source":         "EH",
        "k_units":        "h/Mpc",
        "pk_units":       "(Mpc/h)^3",
        "sigma8_target":  args.sigma8,
        "sigma8_integral_z0": sigma8_check,
        "amplitude_A":    amp,
        "note": (
            "Espectro EH no-wiggle (Eisenstein & Hu 1998, ApJ 496, 605). "
            "Implementación Python independiente del generador Rust. "
            "sigma8_integral_z0 se calcula integrando sobre k=1e-5..500 h/Mpc "
            "a z=0; puede diferir de sigma8_target por la discretización."
        ),
        "bins": bins_eh,
    }

    # ── Intento CAMB (opcional) ─────────────────────────────────────────────
    if args.try_camb:
        bins_camb, camb_info = try_generate_camb_spectrum(
            args.omega_m, args.omega_b, args.h, args.n_s, args.sigma8, args.z,
            k_min=args.k_min, k_max=args.k_max, n_k=args.n_k,
        )
        if bins_camb is not None:
            print(f"  CAMB disponible: {camb_info}")
            result["source"] = "CAMB+EH"
            result["bins_eh"] = bins_eh
            result["bins"] = bins_camb   # CAMB es la referencia principal
            result["camb_info"] = camb_info
        else:
            print(f"  CAMB no disponible: {camb_info}")
            result["camb_status"] = str(camb_info)

    # ── Guardar ─────────────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Guardado: {args.out}  ({len(result['bins'])} bins)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
