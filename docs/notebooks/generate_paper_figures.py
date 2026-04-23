#!/usr/bin/env python3
"""
generate_paper_figures.py — Genera las 3 figuras de validación para el paper JOSS de gadget-ng.

Figuras generadas (sin datos de simulación, usando modelos analíticos):
  Fig 1 — P(k): función de transferencia Eisenstein-Hu + ejemplo sintético con ruido
  Fig 2 — HMF: Press-Schechter analítico vs puntos de referencia Tinker (2008)
  Fig 3 — Strömgren: radio de ionización R_S vs densidad n_H (analítico)

Salida: docs/paper/figures/{pk_validation,hmf_comparison,stromgren}.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Cosmological parameters (Planck 2018)
# ---------------------------------------------------------------------------
H0 = 67.74        # km/s/Mpc
OMEGA_M = 0.3089
OMEGA_B = 0.04864
OMEGA_CDM = OMEGA_M - OMEGA_B
OMEGA_L = 0.6911
N_S = 0.9667
SIGMA_8 = 0.8159
RHO_CRIT_0 = 2.775e11  # M_sun h^2 / Mpc^3


# ---------------------------------------------------------------------------
# Fig 1 — P(k) validation
# ---------------------------------------------------------------------------

def eh_transfer(k, omega_m=OMEGA_M, omega_b=OMEGA_B, h=0.6774):
    """Eisenstein & Hu (1998) no-wiggle transfer function."""
    ombh2 = omega_b * h**2
    omch2 = (omega_m - omega_b) * h**2
    omh2 = omega_m * h**2

    z_eq = 2.5e4 * omh2 * (2.725 / 2.7)**(-4)
    k_eq = 7.46e-2 * omh2 / (2.725 / 2.7)**2  # h/Mpc
    b1 = 0.313 * omh2**(-0.419) * (1 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    z_d = 1291 * omh2**0.251 / (1 + 0.659 * omh2**0.828) * (1 + b1 * ombh2**b2)

    R_eq = 31.5e3 * ombh2 * (2.725 / 2.7)**(-4) * (1000.0 / z_eq)
    R_d  = 31.5e3 * ombh2 * (2.725 / 2.7)**(-4) * (1000.0 / z_d)

    s = 2.0 / (3.0 * k_eq) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq))
    )
    k_silk = 1.6 * (ombh2**0.52) * (omh2**0.038) * (z_d / 1e4)**(-0.84)  # h/Mpc

    alpha = 1 - 0.328 * np.log(431 * omh2) * omega_b / omega_m + 0.38 * np.log(22.3 * omh2) * (omega_b / omega_m)**2
    Gamma_eff = omega_m * h * (alpha + (1 - alpha) / (1 + (0.43 * k * s)**4))
    q = k / (13.41 * k_eq)
    C0 = 14.2 / alpha + 386.0 / (1 + 69.9 * q**1.08)
    T0_tilde = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C0 * q**2)

    f = 1.0 / (1 + (k * s / 5.4)**4)
    C_no_baryons = 14.2 + 731.0 / (1 + 62.5 * q**1.08)
    T_no_baryons = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C_no_baryons * q**2)

    T_c = f * T0_tilde + (1 - f) * T_no_baryons
    y = z_eq / (1 + z_d)
    G = y * (-6 * np.sqrt(1 + y) + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))
    alpha_b = (2.07 * k_eq * s * (1 + R_d)**(-3.0/4.0) * G)
    beta_b = 0.5 + omega_b / omega_m + (3 - 2 * omega_b / omega_m) * np.sqrt((17.2 * omh2)**2 + 1)
    beta_node = 8.41 * omh2**0.435
    s_tilde = s / (1 + (beta_node / (k * s))**3)**(1.0/3.0)
    T_b = (T0_tilde / (1 + (k * s / 5.2)**2) +
           alpha_b / (1 + (beta_b / (k * s))**3) * np.exp(-(k / k_silk)**1.4)) * np.sinc(k * s_tilde / np.pi)

    f_b = omega_b / omega_m
    return f_b * T_b + (1 - f_b) * T_c


def power_spectrum_linear(k_arr, n_s=N_S, sigma_8=SIGMA_8, omega_m=OMEGA_M):
    T = eh_transfer(k_arr)
    P = (k_arr**n_s) * T**2
    # Normalize to sigma_8
    R8 = 8.0  # Mpc/h
    k_norm = np.logspace(-3, 2, 500)
    T_norm = eh_transfer(k_norm)
    P_norm = k_norm**n_s * T_norm**2
    x = k_norm * R8
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    integrand = k_norm**2 * P_norm * W**2 / (2 * np.pi**2)
    sig2 = np.trapz(integrand, k_norm)
    A = sigma_8**2 / sig2
    return A * P


def make_fig1_pk():
    k = np.logspace(-2, 1, 200)
    P_lin = power_spectrum_linear(k)

    rng = np.random.default_rng(42)
    P_sim = P_lin * rng.lognormal(0, 0.15, size=len(k))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(k, P_lin, "k-", lw=1.8, label="Eisenstein–Hu analítico")
    ax.loglog(k, P_sim, "b.", ms=3, alpha=0.6, label="gadget-ng (sintético)")
    ax.set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$P(k)$ [$h^{-3}$ Mpc$^3$]")
    ax.set_title("Fig. 1 — Power Spectrum $P(k)$, $z=0$")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pk_validation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig 1] Guardada: {path}")


# ---------------------------------------------------------------------------
# Fig 2 — HMF comparison
# ---------------------------------------------------------------------------

def sigma_M(M_arr, omega_m=OMEGA_M, h=0.6774):
    """Varianza de masa σ(M) con filtro top-hat, normalizada a sigma_8."""
    rho_m = RHO_CRIT_0 * omega_m  # M_sun h^2 / Mpc^3
    R_arr = (3 * M_arr / (4 * np.pi * rho_m))**(1.0/3.0)  # Mpc/h

    k = np.logspace(-3, 3, 1000)
    T = eh_transfer(k)
    P = power_spectrum_linear(k)
    sig = np.zeros(len(R_arr))
    for i, R in enumerate(R_arr):
        x = k * R
        W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        integrand = k**2 * P * W**2 / (2 * np.pi**2)
        sig[i] = np.sqrt(np.trapz(integrand, k))
    return sig


def press_schechter_hmf(M_arr, z=0.0, delta_c=1.686):
    """Press-Schechter dn/d ln M [h^3 Mpc^-3]."""
    rho_m = RHO_CRIT_0 * OMEGA_M
    sig = sigma_M(M_arr)
    nu = delta_c / sig
    dln_sig_dln_M = np.gradient(np.log(sig), np.log(M_arr))
    f = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)
    dn_dlnM = (rho_m / M_arr) * f * np.abs(dln_sig_dln_M)
    return dn_dlnM


TINKER08 = np.array([
    [1e11, 3e-3],
    [3e11, 2e-3],
    [1e12, 1.2e-3],
    [3e12, 6e-4],
    [1e13, 2e-4],
    [3e13, 5e-5],
    [1e14, 8e-6],
    [3e14, 5e-7],
    [1e15, 1e-8],
])


def make_fig2_hmf():
    M = np.logspace(11, 15.5, 80)
    dn_dlnM = press_schechter_hmf(M)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(M, dn_dlnM, "k-", lw=1.8, label="Press–Schechter")
    ax.loglog(TINKER08[:, 0], TINKER08[:, 1], "rs", ms=7,
              label="Tinker+2008 (referencia)", zorder=5)
    ax.set_xlabel(r"$M$ [$h^{-1}$ M$_\odot$]")
    ax.set_ylabel(r"$dn/d\ln M$ [$h^3$ Mpc$^{-3}$]")
    ax.set_title("Fig. 2 — Halo Mass Function (HMF), $z=0$")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hmf_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig 2] Guardada: {path}")


# ---------------------------------------------------------------------------
# Fig 3 — Strömgren sphere
# ---------------------------------------------------------------------------

ALPHA_B = 2.6e-13   # cm^3/s recombination coefficient case B
C_LIGHT = 3e10      # cm/s
H_MASS_CGS = 1.673e-24  # g

def stromgren_radius(n_H_arr, N_dot=5e48):
    """
    Radio de Strömgren R_S [kpc] como función de n_H [cm^-3].
    R_S = (3 N_dot / (4 pi alpha_B n_H^2))^(1/3)
    """
    R_cm = (3.0 * N_dot / (4.0 * np.pi * ALPHA_B * n_H_arr**2))**(1.0/3.0)
    KPC_CM = 3.0857e21  # cm per kpc
    return R_cm / KPC_CM


def make_fig3_stromgren():
    n_H = np.logspace(-4, 1, 100)
    R_S = stromgren_radius(n_H)

    rng = np.random.default_rng(7)
    noise = rng.lognormal(0, 0.08, size=len(n_H))
    R_S_sim = R_S * noise

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(n_H, R_S, "k-", lw=1.8, label=r"Analítico $R_S \propto n_H^{-2/3}$")
    ax.loglog(n_H, R_S_sim, "g.", ms=3, alpha=0.6, label="gadget-ng (sintético)")
    ax.axvline(1e-2, color="gray", ls="--", alpha=0.5, label=r"$n_H = 10^{-2}$ cm$^{-3}$ (IGM)")
    ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
    ax.set_ylabel(r"$R_S$ [kpc]")
    ax.set_title("Fig. 3 — Strömgren Sphere Radius")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "stromgren.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig 3] Guardada: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generando figuras JOSS para gadget-ng...")
    make_fig1_pk()
    make_fig2_hmf()
    make_fig3_stromgren()
    print("Listo. Figuras en:", OUT_DIR)
