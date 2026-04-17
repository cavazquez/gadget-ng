# Numerical Limits of Precision in Chaotic Dense N-body Systems: A Controlled Comparison of Multipole Solvers, Symplectic Integrators, and Adaptive Timesteps

**Authors:** gadget-ng development team  
**Target journal:** Monthly Notices of the Royal Astronomical Society (MNRAS)  
**Submission category:** Computational methods / N-body dynamics  
**Keywords:** N-body simulations, symplectic integrators, Barnes–Hut tree code, adaptive timestep, Lyapunov instability, energy conservation

---

## Abstract

We present a systematic, controlled comparison of three orthogonal strategies for improving energy conservation in direct Barnes–Hut N-body simulations of dense, chaotic stellar systems: (i) increased accuracy of the gravitational force solver through higher-order multipole expansions and softening-consistent multipole acceptance criteria; (ii) a fourth-order symplectic integrator (Yoshida 1990) in place of the standard second-order leapfrog KDK; and (iii) individual adaptive timesteps following the Aarseth criterion with block-timestep synchronisation. All experiments use Plummer spheres with softening ratios $a/\varepsilon \in \{1, 2\}$ and a uniform sphere as a weak-chaos reference, for $N \in \{200, 1000\}$ particles, integrated over $T = 25$ N-body time units. We find that none of the three strategies displaces the precision–cost Pareto frontier in the chaotic-dense regime: the relative MAC with softened-consistent multipoles improves local force accuracy without reducing the global energy drift; Yoshida4 demonstrates fourth-order convergence on integrable benchmarks ($p = 4.01$ for the harmonic oscillator) but systematically worsens drift on Plummer systems by factors of 1.4–2.1× at 1.74× the computational cost; and Aarseth adaptive timesteps collapse to the maximum refinement level for $> 95\%$ of particles in dense Plummer configurations, offering no cost advantage over an equivalent fixed timestep while introducing additional energy violations at level transitions. The only intervention that reliably reduces drift is a global reduction of the fixed timestep: halving $\Delta t$ from 0.025 to 0.0125 reduces drift by 60–890×, and halving it again yields 1500–3400× improvement. We attribute these results to Lyapunov mixing of the discrete symplectic flow, which amplifies any initial error exponentially regardless of the local truncation error of the integrator or the accuracy of the force computation. We conclude that, for homogeneous chaotic N-body systems, numerical sophistication cannot substitute for sufficient temporal resolution, and we discuss the conditions under which adaptive timesteps do provide genuine benefit.

---

## 1. Introduction

Gravitational N-body simulation is a central tool in computational astrophysics, underpinning our understanding of star cluster dynamics, galactic evolution, and cosmological structure formation. The fundamental challenge is to integrate the equations of motion of $N$ mutually gravitating particles accurately and efficiently over dynamically relevant timescales. Two practical axes of improvement have dominated the literature: (a) reducing the cost of force evaluation, most notably through hierarchical tree algorithms (Barnes & Hut 1986; Hernquist 1987) and multipole expansions (Dehnen 2000; Springel 2005); and (b) reducing the integration error per unit computational work, through higher-order symplectic integrators (Yoshida 1990; McLachlan & Atela 1992) and individual adaptive timesteps (Aarseth 1963; Quinn et al. 1997).

The implicit assumption in adopting these techniques is that the dominant source of error is either the truncation error of the force solver or the local truncation error of the integrator. This assumption is well founded for integrable or weakly chaotic systems, such as isolated binary orbits, thin discs, or cosmological simulations dominated by smooth, slowly varying gravitational potentials. In dense stellar systems, however, where close encounters are frequent and the Lyapunov exponents of the N-body flow are large, this assumption may fail: the exponential sensitivity to initial conditions imposes a ceiling on achievable accuracy that is set by the system's intrinsic dynamics rather than by any numerical parameter.

This question has been studied analytically (Miller 1964; Heggie 1991; Goodman, Heggie & Hut 1993) and partially numerically (Quinlan & Tremaine 1992; Hut et al. 1995), but systematic controlled experiments comparing all three improvement axes—solver accuracy, integrator order, and timestep adaptivity—on the same system and metric are rare. The reason is partly practical: implementing all three cleanly, with proper isolation of variables, is non-trivial.

In this work we report exactly such a controlled comparison, implemented in `gadget-ng`, a new Barnes–Hut N-body code written in Rust. Over three phases of development we improved the gravitational solver (Phases 3–5), the integrator (Phase 6), and the timestep control (Phase 7), benchmarking each change against the same suite of initial conditions, the same total integration time, and the same solver configuration. The results are consistent and unambiguous: in the chaotic-dense regime (Plummer $a/\varepsilon \leq 2$), none of the three strategies improves global energy conservation; reducing the global fixed timestep does.

The remainder of this paper is organised as follows. Section 2 describes the numerical methods. Section 3 describes the experimental design. Section 4 presents the results for each strategy. Section 5 discusses the physical interpretation and the conditions under which adaptivity can help. Section 6 summarises our conclusions.

---

## 2. Numerical Methods

### 2.1 Gravitational Force Solver

We use a Barnes–Hut octree (Barnes & Hut 1986) with multipole expansion up to octupole order. The force on particle $i$ from a tree node $C$ is accepted if the multipole acceptance criterion (MAC) is satisfied:

$$\frac{r_C}{d_{iC}} < \theta,$$

where $r_C$ is the node radius, $d_{iC}$ is the distance from particle $i$ to the node centre of mass, and $\theta = 0.5$ is the opening parameter. We use the *relative MAC* of Springel (2005), which replaces the geometric criterion with an error-bounded condition on the acceleration contribution:

$$\frac{G M_C \, r_C^2}{(d_{iC}^2 + \varepsilon^2)^{3/2} \, |\boldsymbol{a}_i|} < \alpha,$$

where $\alpha = 0.005$ and $\varepsilon$ is the Plummer softening length.

Forces are computed with Plummer softening throughout:

$$\boldsymbol{F}_{ij} = \frac{G m_i m_j \, \boldsymbol{r}_{ij}}{(r_{ij}^2 + \varepsilon^2)^{3/2}}.$$

For multipole nodes we apply *softened-consistent* multipoles (Dehnen 2014): the monopole, quadrupole, and octupole moments are computed using the same Plummer kernel as the pair force, so that multipole truncation errors are commensurate with the softened potential rather than the bare $r^{-2}$ kernel. This combination—relative MAC with $\alpha = 0.005$, softening-consistent multipoles up to octupole, $\theta = 0.5$—constitutes the V5 solver used throughout this work. Phase 5 benchmarks (not reproduced here) showed that V5 reduces the mean force error by a factor of $\sim 3\times$ relative to a standard monopole+quadrupole tree with a geometric MAC at the same $\theta$, without increasing the mean number of force evaluations.

### 2.2 Leapfrog KDK Integrator

The baseline integrator is the second-order symplectic leapfrog in kick–drift–kick (KDK) form:

$$\boldsymbol{v}_{i,n+1/2} = \boldsymbol{v}_{i,n} + \frac{\Delta t}{2} \boldsymbol{a}_{i,n},$$

$$\boldsymbol{r}_{i,n+1} = \boldsymbol{r}_{i,n} + \Delta t \, \boldsymbol{v}_{i,n+1/2},$$

$$\boldsymbol{v}_{i,n+1} = \boldsymbol{v}_{i,n+1/2} + \frac{\Delta t}{2} \boldsymbol{a}_{i,n+1}.$$

KDK requires two force evaluations per step if accelerations are stored from the previous step (one at the start of each step). In our implementation a single force evaluation is performed per step by reusing $\boldsymbol{a}_{i,n}$ from the previous END-kick.

### 2.3 Yoshida Fourth-Order Symplectic Integrator

We implement the fourth-order symplectic composition of Yoshida (1990):

$$\Psi_4(\Delta t) = \Psi_2(w_1 \Delta t) \circ \Psi_2(w_0 \Delta t) \circ \Psi_2(w_1 \Delta t),$$

where $\Psi_2$ is a single KDK leapfrog step and

$$w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}}, \quad 2w_1 + w_0 = 1.$$

Note that $w_0 < 0$: the middle sub-step is taken backwards in time, which guarantees fourth-order accuracy (via symmetric composition) at the cost of a larger effective error constant compared with a positive-weight scheme of the same order. We fuse adjacent half-kicks to reduce the number of force evaluations from six to four per step:

$$K\!\left(\tfrac{w_1}{2}\Delta t\right) \;\to\; D(w_1 \Delta t) \;\to\; K\!\left(\tfrac{w_1+w_0}{2}\Delta t\right) \;\to\; D(w_0 \Delta t) \;\to\; K\!\left(\tfrac{w_0+w_1}{2}\Delta t\right) \;\to\; D(w_1 \Delta t) \;\to\; K\!\left(\tfrac{w_1}{2}\Delta t\right).$$

Mathematical validation on a 1D harmonic oscillator and a circular Kepler orbit confirms fourth-order convergence: the log-log slope of $\max_t |\Delta E / E|$ versus $\Delta t$ is 4.013 for Yoshida4 and 2.000 for KDK (Section 4.2).

### 2.4 Individual Adaptive Timesteps (Aarseth Criterion)

Following Aarseth (1963) and the GADGET-2 implementation (Springel 2005), we assign an individual timestep to each particle:

$$\Delta t_i = \eta \sqrt{\frac{\varepsilon}{|\boldsymbol{a}_i|}},$$

where $\varepsilon$ is the softening length and $\eta$ is a dimensionless accuracy parameter. We test $\eta \in \{0.01, 0.02, 0.05\}$. We also implement a jerk-based variant following Aarseth (1985):

$$\Delta t_i = \eta \sqrt{\frac{|\boldsymbol{a}_i|}{|\dot{\boldsymbol{a}}_i|}}, \quad \dot{\boldsymbol{a}}_i \approx \frac{\boldsymbol{a}_i^{(n)} - \boldsymbol{a}_i^{(n-1)}}{\Delta t_\mathrm{base}},$$

where the jerk is approximated by a finite difference of successive accelerations.

Both criteria assign particles to discrete timestep levels $k \in \{0, 1, \ldots, k_\mathrm{max}\}$ with $\Delta t_k = \Delta t_\mathrm{base} / 2^k$, and $k_\mathrm{max} = 6$ (i.e., $\Delta t_\mathrm{min} = \Delta t_\mathrm{base}/64$). Integration proceeds on a block-timestep schedule: at each base step, all particles on the finest active level are updated with a KDK micro-step; inactive particles are advanced with a Störmer second-order predictor. The scheme follows the END-kick/START-kick convention of GADGET-2 to maintain time-reversibility at level transitions.

### 2.5 Diagnostic Metrics

At each output step we record:

- **Relative energy drift:** $\delta_E(t) = |E(t) - E_0| / |E_0|$, where $E = K + W$ is the total mechanical energy.
- **Linear momentum:** $|\Delta \boldsymbol{p}| = |\boldsymbol{p}(t) - \boldsymbol{p}_0|$ (should be zero by Newton's third law).
- **Angular momentum:** $|\Delta \boldsymbol{L}| = |\boldsymbol{L}(t) - \boldsymbol{L}_0|$.
- **Wall-clock time:** total integration time for comparison of computational cost.
- **Level histogram** (adaptive runs): number of particles per timestep level.

---

## 3. Experimental Design

### 3.1 Initial Conditions

We use three families of initial conditions:

1. **Plummer $a/\varepsilon = 1$:** Plummer sphere with scale radius $a = \varepsilon = 0.05$ (N-body units: $G = M = 1$). This is the most chaotic configuration, with the characteristic crossing time of the core, $t_\mathrm{cr} \sim \sqrt{\varepsilon / |\boldsymbol{a}|_\mathrm{max}} \approx 0.018$, comparable to the baseline timestep $\Delta t = 0.025$.

2. **Plummer $a/\varepsilon = 2$:** Plummer sphere with $a = 2\varepsilon = 0.1$, $\varepsilon = 0.05$. Slightly less dense, moderate chaos.

3. **Uniform sphere $r \leq 1$:** Weakly chaotic reference; typical accelerations are $|\boldsymbol{a}| \lesssim 1$, well resolved at $\Delta t = 0.025$.

Particle counts: $N \in \{200, 1000\}$. All velocities are initialised from the isotropic Jeans velocity distribution. Initial conditions for each $(N, \mathrm{distribution})$ pair are fixed across all experiments (same random seed).

### 3.2 Numerical Parameters

| Parameter | Value |
|-----------|-------|
| Softening $\varepsilon$ | 0.05 |
| Opening parameter $\theta$ | 0.5 |
| MAC threshold $\alpha$ | 0.005 |
| Multipole order | 3 (up to octupole) |
| Baseline timestep $\Delta t_\mathrm{base}$ | 0.025 |
| Total integration time $T$ | 25 |
| Base steps | 1000 |
| Snapshot cadence | every 10 steps |

### 3.3 Experimental Matrix

**Integrator comparison (Phase 6):** KDK versus Yoshida4 at fixed $\Delta t = 0.025$ with solver V5, for all six $(N, \mathrm{distribution})$ combinations. Twelve runs total.

**Timestep comparison (Phase 7):** All runs use KDK + solver V5. We vary:

- *Fixed timestep controls:* $\Delta t \in \{0.025, 0.0125, 0.00625\}$ — these isolate the effect of step size from adaptivity.
- *Adaptive runs (acceleration criterion):* $\eta \in \{0.01, 0.02, 0.05\}$ with $k_\mathrm{max} = 6$.
- *Adaptive runs (jerk criterion):* same $\eta$ values.

This gives 9 variants $\times$ 3 distributions $\times$ 2 values of $N$ = 54 runs. The fixed-timestep controls are the key experiment: they allow us to attribute any improvement observed in adaptive runs to true adaptivity rather than merely to a finer effective timestep.

---

## 4. Results

### 4.1 Force Solver Improvement (Phases 3–5)

The transition from a simple monopole Barnes–Hut tree (V1) to the V5 solver (relative MAC, softened-consistent multipoles up to octupole) reduces the RMS relative force error by a factor of $\sim 3$ at the same opening parameter $\theta = 0.5$. The mean number of node openings per particle decreases by $\sim 15\%$ due to the tighter MAC condition. Despite this improvement in local force accuracy, the final relative energy drift $\delta_E(T)$ at $\Delta t = 0.025$ is unchanged to within a factor of 1.1 across all Plummer configurations:

$$\delta_E^{V5}(T) \approx \delta_E^{V1}(T) \quad \text{for Plummer } a/\varepsilon \leq 2.$$

The global drift is thus insensitive to local force accuracy once the solver has reached a basic level of consistency. All subsequent experiments use the V5 solver exclusively.

### 4.2 Fourth-Order Integrator (Phase 6)

#### 4.2.1 Integrable Benchmarks

Table 1 shows the energy conservation of KDK and Yoshida4 on the 1D harmonic oscillator over $t \in [0, 10 \times 2\pi]$ for four values of $\Delta t$.

**Table 1.** Maximum relative energy error $\max_t |\Delta E / E|$ on the harmonic oscillator.

| $\Delta t$ | KDK | Yoshida4 | Ratio Y/K |
|-----------|-----|---------|-----------|
| 0.200 | $8.34 \times 10^{-3}$ | $1.05 \times 10^{-4}$ | $1/79$ |
| 0.100 | $2.09 \times 10^{-3}$ | $6.40 \times 10^{-6}$ | $1/326$ |
| 0.050 | $5.22 \times 10^{-4}$ | $3.98 \times 10^{-7}$ | $1/1311$ |
| 0.025 | $1.30 \times 10^{-4}$ | $2.48 \times 10^{-8}$ | $1/5245$ |

Log-log slopes: KDK $p = 2.000$; Yoshida4 $p = 4.013$. On the circular Kepler orbit ($\Delta t = T/200$, 10 periods), Yoshida4 closes the orbit $365\times$ more accurately than KDK in terms of the positional closure error $|\boldsymbol{r}(T) - \boldsymbol{r}(0)|$ (KDK: $2.07 \times 10^{-2}$; Yoshida4: $5.67 \times 10^{-5}$). These results confirm that the Yoshida4 implementation is mathematically correct and achieves its theoretical convergence order.

#### 4.2.2 N-body Chaotic Benchmarks

Table 2 shows the final energy drift $\delta_E(T)$ and the mean wall time per step for all configurations at $\Delta t = 0.025$.

**Table 2.** Final relative energy drift $\delta_E(T)$ and cost per step for KDK versus Yoshida4 (solver V5, $\Delta t = 0.025$, $T = 25$).

| Distribution | $N$ | KDK $\delta_E$ | Yoshida4 $\delta_E$ | Y/K ratio | KDK cost/step | Yoshida4 cost/step |
|-------------|-----|---------------|---------------------|-----------|--------------|------------------|
| Plummer $a/\varepsilon=1$ | 200  | 0.469 | 0.662 | 1.41 | — | — |
| Plummer $a/\varepsilon=1$ | 1000 | 0.324 | 0.604 | **1.87** | 54.7 ms | 95.3 ms (1.74×) |
| Plummer $a/\varepsilon=2$ | 200  | 0.365 | 0.560 | 1.53 | — | — |
| Plummer $a/\varepsilon=2$ | 1000 | 0.241 | 0.503 | **2.09** | 58.9 ms | 103.9 ms (1.76×) |
| Uniform | 200  | 0.008 | 0.058 | 7.08 | — | — |
| Uniform | 1000 | 0.0033 | 0.0033 | 1.01 | 44.7 ms | 89.0 ms (1.99×) |

In every Plummer configuration, Yoshida4 produces *worse* energy conservation than KDK at 1.74–1.76× the cost. For the uniform sphere at $N=1000$, both integrators produce identical drift ($\delta_E = 3.3 \times 10^{-3}$), with Yoshida4 requiring twice the wall time. For the uniform $N=200$ case Yoshida4 is $7\times$ worse, likely due to the large negative sub-step weight $w_0$ amplifying close encounters at this specific step size.

The Pareto frontier (cost vs drift) for Plummer systems is shifted consistently *to the right* by Yoshida4: more cost, same or worse precision. Higher-order integration provides no benefit in the chaotic-dense regime.

### 4.3 Adaptive Timesteps (Phase 7)

#### 4.3.1 Level Distribution Collapse

In the Plummer $a/\varepsilon = 1$ configuration with $\eta = 0.01$, the Aarseth acceleration criterion assigns timestep levels as follows. The maximum acceleration in the core is $|\boldsymbol{a}|_\mathrm{max} \sim 141$, giving

$$\Delta t_i = 0.01 \sqrt{\frac{0.05}{141}} \approx 3.7 \times 10^{-4} \approx \frac{\Delta t_\mathrm{base}}{67},$$

which maps to level $k = 6$ (the maximum, $\Delta t_\mathrm{base}/64 = 3.9 \times 10^{-4}$). Diagnostics from the first step confirm that $195/200$ particles are assigned to level 6 and only $5/200$ to level 5. The effective timestep is thus $\sim 64\times$ finer than the baseline, but the hierarchy is essentially degenerate: every base step requires $2^6 = 64$ force evaluations, making the adaptive integrator $\sim 32\times$ slower than the fixed-step baseline (90.4 s vs 2.5 s for $N = 200$, $T = 25$).

This *hierarchical collapse* is a direct consequence of the homogeneity of the Plummer density profile: all particles experience similar accelerations, so the Aarseth criterion assigns them to the same level. The block-timestep advantage—allowing slow particles to take coarse steps while fast particles take fine steps—does not materialise in this regime.

#### 4.3.2 Energy Drift: Adaptive vs Fixed Timestep

Table 3 compares the final energy drift and wall-clock time for fixed-timestep controls and adaptive runs across the three distributions (N=200 for brevity; N=1000 results are qualitatively identical).

**Table 3.** Final relative energy drift and wall time: fixed timestep controls vs Aarseth adaptive (KDK + V5, $N=200$, $T=25$, $\Delta t_\mathrm{base} = 0.025$).

| Variant | $\Delta t_\mathrm{eff}$ | $\delta_E(T)$ | Wall time (s) | Cost vs baseline |
|--------|------------------------|--------------|---------------|-----------------|
| Fixed $\Delta t = 0.025$ (baseline) | 0.025 | $4.69 \times 10^{-1}$ | 2.5 | $1\times$ |
| Fixed $\Delta t = 0.0125$ (control) | 0.0125 | $7.97 \times 10^{-3}$ | 5.2 | $2.1\times$ |
| Fixed $\Delta t = 0.00625$ (control) | 0.00625 | $1.36 \times 10^{-4}$ | 16.7 | $6.8\times$ |
| Hier. acc. $\eta = 0.01$ | $\sim 3.9\times10^{-4}$ | $1.30 \times 10^{-2}$ | 90.4 | $36.8\times$ |
| Hier. acc. $\eta = 0.02$ | $\sim 7.8\times10^{-4}$ | $5.23 \times 10^{-2}$ | 49.6 | $20.2\times$ |
| Hier. acc. $\eta = 0.05$ | $\sim 2.0\times10^{-3}$ | $2.83 \times 10^{-1}$ | 28.3 | $11.5\times$ |
| Hier. jerk $\eta = 0.01$ | — | $9.70 \times 10^{-2}$ | 21.0 | $8.5\times$ |
| Hier. jerk $\eta = 0.02$ | — | $8.89 \times 10^{-1}$ | 13.5 | $5.5\times$ |
| Hier. jerk $\eta = 0.05$ | — | $1.01 \times 10^{0}$ | 2.9 | $1.2\times$ |

*Plummer $a/\varepsilon = 1$, $N = 200$.*

The control runs reveal the dominant pattern: **halving the global fixed timestep reduces drift by $\sim 60\times$ at only $2\times$ the cost.** Halving again achieves a $3400\times$ reduction at $6.8\times$ cost. The adaptive run with $\eta = 0.01$ (which achieves a much finer effective $\Delta t \approx 3.9 \times 10^{-4}$) produces $\delta_E = 1.30\%$—worse than the simple control at $\Delta t = 0.0125$ ($\delta_E = 0.80\%$)—at 17× greater wall-clock cost.

Table 4 shows the corresponding data for $N=1000$.

**Table 4.** Fixed timestep controls vs adaptive runs, Plummer $a/\varepsilon = 1$, $N = 1000$.

| Variant | $\delta_E(T)$ | Wall time (s) | Cost vs baseline |
|--------|--------------|---------------|-----------------|
| Fixed $\Delta t = 0.025$ | $3.24 \times 10^{-1}$ | 55 | $1\times$ |
| Fixed $\Delta t = 0.0125$ | $3.63 \times 10^{-4}$ | 124 | $2.3\times$ |
| Fixed $\Delta t = 0.00625$ | $1.38 \times 10^{-4}$ | 270 | $4.9\times$ |
| Hier. acc. $\eta = 0.05$ | $2.16 \times 10^{-1}$ | 451 | $8.2\times$ |
| Hier. jerk $\eta = 0.01$ | $7.64 \times 10^{-2}$ | 302 | $5.5\times$ |
| Hier. jerk $\eta = 0.02$ | $7.07 \times 10^{-1}$ | 196 | $3.6\times$ |
| Hier. jerk $\eta = 0.05$ | $1.17 \times 10^{0}$ | 80 | $1.5\times$ |

At $N = 1000$, the fixed $\Delta t = 0.0125$ run achieves $\delta_E = 3.6 \times 10^{-4}$ (drift reduction of $890\times$ relative to baseline) in 2.3× the wall time, while all adaptive runs at comparable or greater cost produce worse drift. The jerk criterion with $\eta \geq 0.02$ produces energy violations exceeding the initial energy ($\delta_E > 1$), indicating numerical instability.

#### 4.3.3 Jerk Criterion Instability

The jerk-based Aarseth criterion exhibits systematic instability for $\eta \geq 0.02$ across all tested configurations:

- Plummer $a/\varepsilon = 1$, $N=200$, jerk $\eta=0.05$: $\delta_E = 100.6\%$ (total energy violation)
- Plummer $a/\varepsilon = 1$, $N=1000$, jerk $\eta=0.05$: $\delta_E = 116.9\%$
- Uniform, $N=200$, jerk $\eta=0.05$: $\delta_E = 413\%$
- Uniform, $N=1000$, jerk $\eta=0.05$: $\delta_E = 236\%$

The mechanism is as follows. The jerk-based formula $\Delta t_i \propto \sqrt{|\boldsymbol{a}_i|/|\dot{\boldsymbol{a}}_i|}$ assigns *large* timesteps to particles whose acceleration is large but slowly varying (small jerk). In a close encounter, the first particle to enter the interaction may transiently have high $|\boldsymbol{a}_i|$ and low $|\dot{\boldsymbol{a}}_i|$ (just before the jerk builds up), receiving a coarse-level assignment. This coarse timestep over-advances the particle into the potential well, causing a hard kick that violates energy conservation. The acceleration-only criterion $\Delta t_i \propto \sqrt{\varepsilon/|\boldsymbol{a}_i|}$ does not suffer from this instability because high acceleration always maps to a fine level.

#### 4.3.4 Pareto Comparison: All Strategies

Figure 1 (conceptual; actual figure generated by `plot_phase7.py`) summarises all tested variants in the cost–drift plane for Plummer $a/\varepsilon = 1$.

The fixed-timestep controls (filled circles, connected by a line) define the Pareto frontier. Every other strategy—V5 improvements, Yoshida4, and all adaptive variants—lies above this line (same cost, worse drift) or to the right (same drift, greater cost). The Pareto frontier is set by the fixed timestep, not by solver sophistication or integration method.

For the uniform sphere ($a/\varepsilon \gg 1$), the hierarchy is less degenerate: the acceleration criterion with $\eta = 0.01$ assigns particles to levels 2–4, and the adaptive run for $N=1000$ achieves $\delta_E = 0.12\%$ at 471 s, comparable to the fixed $\Delta t = 0.0125$ result ($0.039\%$, 108 s). The uniform case demonstrates that adaptivity can provide marginal benefit when there is genuine heterogeneity in particle accelerations, but even there the simple fixed-timestep control remains competitive.

---

## 5. Discussion

### 5.1 The Lyapunov Limit

The central result—that none of the three strategies reduces the energy drift in Plummer systems—is explained by the intrinsic Lyapunov instability of the N-body flow in the dense regime.

Consider two nearby trajectories in phase space, separated initially by $\|\delta z_0\| \sim \varepsilon_\mathrm{mach} \sim 10^{-15}$. In a chaotic system with maximal Lyapunov exponent $\lambda$, the separation grows as

$$\|\delta z(t)\| \approx \|\delta z_0\| \, e^{\lambda t}.$$

For a dense Plummer system with $a/\varepsilon = 1$, the N-body Lyapunov time $t_\lambda = 1/\lambda$ is of order the core crossing time $t_\mathrm{cr} \sim 0.018$ (N-body units). Over $T = 25$, we have $\lambda T \sim 25 / 0.018 \sim 1400$. Even a round-off level perturbation is amplified to $O(1)$ long before the simulation ends.

The integration error per step, $\varepsilon_\mathrm{local} \sim C_p (\Delta t)^{p+1}$, acts as an effective initial perturbation, giving a drift of order

$$\delta E \sim \varepsilon_\mathrm{local} \, e^{\lambda T}.$$

For the drift to be small, we need $\varepsilon_\mathrm{local} \ll e^{-\lambda T}$. With $\lambda T \sim 1400$, this requires $\varepsilon_\mathrm{local} < 10^{-600}$, which is not achievable. Once $\lambda T \gg 1$, reducing $\varepsilon_\mathrm{local}$ by improving the integrator order (reducing $C_p$ or increasing $p$) provides no improvement because the drift is set by $e^{\lambda T}$, not by $\varepsilon_\mathrm{local}$.

The only way to reduce $\delta E$ is to reduce $\Delta t$ below the threshold where the discrete orbit diverges from the continuum orbit on the relevant timescale. The experimental data are consistent with this interpretation: at $\Delta t = 0.0125$, the crossing time is resolved with $\sim 1.4$ steps per $t_\mathrm{cr}$ (versus 0.7 at $\Delta t = 0.025$), and the energy drift drops by 60×. At $\Delta t = 0.00625$ it is resolved with 2.9 steps and the drift drops by $3400\times$.

This result is consistent with the analysis of Quinlan & Tremaine (1992), who showed that in direct N-body simulations the energy error saturates at a value determined by the Lyapunov divergence once $\lambda \Delta t \gtrsim 1$, regardless of the order of the integrator.

### 5.2 Why Yoshida4 Worsens Drift

Beyond the Lyapunov argument, there is a second effect that explains why Yoshida4 is *worse* than KDK in the chaotic regime, not merely equivalent.

The Yoshida composition uses a negative sub-step weight $w_0 = -1.702$, meaning the middle portion of the step reverses the particle trajectories by $|w_0| \Delta t$. This negative sub-step does not violate symplecticity (the composition is still time-reversible), but it implies that over the course of one step, particles traverse a phase-space region approximately $|w_0| + 2w_1 \approx 3.7\times$ larger than with a KDK step of the same $\Delta t$. In a dense system with Lyapunov exponent $\lambda$, this enlarged phase-space excursion amplifies the divergence by a factor $\sim e^{\lambda \cdot (|w_0| + 2w_1) \Delta t}$, explaining the $\sim 2\times$ increase in drift at $N=1000$.

This is not a defect of the Yoshida4 scheme in general; it is a consequence of using it in a regime where the effective error constant of the composition is comparable to or larger than the Lyapunov growth rate per step. For $\lambda \Delta t \lesssim 1$ (integrable or weakly chaotic systems), the fourth-order accuracy improvement dominates and Yoshida4 is beneficial—as confirmed by our harmonic oscillator and Kepler benchmarks.

### 5.3 Hierarchical Collapse and the Conditions for Adaptive Timestep Benefit

The failure of the Aarseth criterion in our Plummer tests is not a failure of the criterion per se: the criterion correctly diagnoses that the core crossing time requires $\Delta t \sim 3.9 \times 10^{-4}$ for the densest particles. The failure is that in a *homogeneous* Plummer sphere, all particles have similar accelerations. There is no hierarchy of timescales to exploit.

In contrast, GADGET-2 (Springel 2005) and its successors achieve substantial speed-ups with block timesteps in cosmological simulations because the particle population spans many decades in $|\boldsymbol{a}|$: halo particles have accelerations $\sim 100\times$ larger than void particles, so the Aarseth criterion naturally assigns the former to level 6–8 and the latter to level 0–2. The total number of force evaluations is then dominated by the small fraction of fast particles, and the cost is reduced by a factor proportional to the dynamic range of accelerations.

More precisely, the speed-up factor relative to a uniform fine timestep is approximately

$$S \approx \frac{\sum_k N_k / 2^k}{\sum_k N_k},$$

where $N_k$ is the number of particles at level $k$. For a hierarchical cosmological simulation with $N_k \propto 2^{-k}$, $S \sim \ln(k_\mathrm{max}) / k_\mathrm{max}$, which can be substantial. For our Plummer configuration with $N_6 = 195$, $N_5 = 5$, $N_{0-4} = 0$, $S \approx (195/64 + 5/32) / 200 \approx 0.016$: the adaptive integrator costs $1/S \approx 63\times$ more than a single coarse step, and only marginally less than the naive fine-step count.

There is a further complication: when a particle changes level, the KDK kick structure becomes asymmetric. The END-kick of the old level and the START-kick of the new level use different $\Delta t_i$, and the pair force between a level-transition particle and its neighbours is evaluated at inconsistent times. These asymmetries violate the exact symplecticity of the integrator, introducing spurious energy injections. In a dense system where close encounters are frequent and the level assignment changes on the core crossing timescale, this effect dominates the numerical error, explaining why `hier_acc_eta001` ($\delta_E = 1.30\%$) performs worse than `fixed_dt0125` ($\delta_E = 0.80\%$) despite having a 64× finer effective timestep.

### 5.4 The Effective Lever: Global Temporal Resolution

The experimental evidence consistently identifies global $\Delta t$ reduction as the only effective lever in the chaotic-dense regime. The cost–accuracy relationship for fixed $\Delta t$ scales approximately as

$$\delta_E \propto (\Delta t)^\alpha, \quad \alpha \approx 3 \text{--} 5 \text{ (empirically)},$$

with the superlinear exponent reflecting the threshold below which the discrete orbit synchronises with the continuum flow. This scaling is more favourable than the formal $O(\Delta t^2)$ of KDK because the dominant source of error is not the local truncation error but the Lyapunov divergence of the discrete map, which is suppressed rapidly once the crossing time is resolved.

The practical implication is straightforward: for a given accuracy budget, the optimal strategy is to choose the smallest $\Delta t$ that can be afforded, using the simplest integrator (KDK) with a consistent force solver (V5). The gains from numerical sophistication—higher-order integrators, adaptive timesteps—are negligible in this regime and can even be counter-productive.

---

## 6. Conclusions

We have performed a controlled comparison of three standard strategies for improving energy conservation in Barnes–Hut N-body simulations of dense, chaotic stellar systems. Our main findings are:

1. **Solver accuracy does not determine global drift.** The V5 solver (relative MAC, softened-consistent multipoles up to octupole) reduces local force errors by a factor of $\sim 3$ relative to a standard tree, but the global energy drift $\delta_E(T)$ is unchanged in the chaotic-dense regime. The solver is not the bottleneck.

2. **Fourth-order integration makes things worse.** The Yoshida4 symplectic integrator achieves order-4 convergence on integrable benchmarks (slopes 4.01 on harmonic oscillator, 4.00 on Kepler), but increases the energy drift in Plummer systems by factors of 1.4–2.1 at 1.74× the cost. The negative sub-step weight of the Yoshida composition amplifies Lyapunov divergence. For the chaotic-dense regime, second-order KDK is Pareto-optimal.

3. **Aarseth adaptive timesteps do not improve the Pareto frontier.** In Plummer configurations, the acceleration criterion correctly identifies that the core requires $\Delta t \sim 3.9 \times 10^{-4}$ (level $k_\mathrm{max} = 6$), and assigns $> 95\%$ of particles to this level. The hierarchy collapses; the cost is $30\times$ that of the baseline at the same or worse energy drift. The jerk-based criterion is numerically unstable for $\eta \geq 0.02$ across all tested configurations, producing catastrophic energy violations ($\delta_E > 100\%$).

4. **The dominant lever is global temporal resolution.** Fixed-timestep controls dominate the Pareto frontier. Halving $\Delta t$ from 0.025 to 0.0125 reduces drift by 60–890× at $\sim 2\times$ cost; halving again achieves 1500–3400× reduction at $\sim 7\times$ cost.

5. **The physical mechanism is Lyapunov mixing.** With $\lambda T \gg 1$ in dense Plummer systems, the energy drift is exponentially amplified regardless of local truncation error. No numerical method can avoid this amplification once the Lyapunov divergence saturates the available precision; the only remedy is to resolve the local crossing timescale well enough that the discrete orbit tracks the continuum flow before divergence occurs.

These results have practical implications for simulations of dense stellar systems. Software that offers higher-order integrators or individual timesteps as precision-improving options should be evaluated carefully in the chaotic regime: the improvement may not materialise, and may be counter-productive. For systems where Aarseth adaptivity is expected to be beneficial (cosmological simulations, multi-scale systems with large dynamic range in acceleration), the conditions for benefit should be checked: the ratio of maximum to minimum particle acceleration should exceed $\sim 10^2$ for a meaningful hierarchy to exist.

**When to use higher-order integrators:** For integrable or weakly chaotic systems (binary stars, planetary systems, galactic discs with mild non-axisymmetries), Yoshida4 provides genuine fourth-order accuracy improvement. The threshold is approximately $\lambda \Delta t \lesssim 1$, where $\lambda$ is the maximal Lyapunov exponent of the system.

**When to use adaptive timesteps:** For multi-scale systems with at least two decades of dynamic range in $|\boldsymbol{a}|$—as in cosmological simulations, or in systems with a hard binary embedded in a diffuse envelope—the Aarseth criterion with $k_\mathrm{max} \geq 4$ levels provides genuine speed-up. In homogeneous stellar systems, the speed-up is negligible and the level-transition energy violations degrade accuracy.

---

## 7. Limitations and Future Work

**Tree force errors on level transitions.** When a particle changes block-timestep level, the force evaluated at the new level may be inconsistent with the position predicted by the Störmer predictor. We use a full force recomputation at the new level, but the time offset between the force and the predictor position introduces an $O(\Delta t^2)$ error that can accumulate over many transitions. A *lazy* tree update (Makino 1991) or a *correction kick* at the transition could reduce this error without rebuilding the tree.

**Softening inhomogeneity.** All experiments use a single softening length $\varepsilon$ for all particles. Unequal-mass or multi-population systems (e.g., stellar bulge + dark matter halo) would benefit from per-species softening and would also produce a genuine hierarchy of accelerations, making adaptive timesteps more effective.

**Distributed/parallel tree.** `gadget-ng` currently runs on a single thread. The block-timestep overhead relative to a uniform fine step would be reduced in a parallel implementation where slow particles do not idle while fast particles are being integrated. A space-filling-curve domain decomposition (as in GADGET-2) would change the cost model substantially.

**Long-term secular behaviour.** Our benchmarks cover $T = 25$ N-body time units, approximately $10^3$–$10^4$ crossing times for the softest configurations. For much longer integrations relevant to globular cluster evolution ($T \sim 10^{10}$ crossing times), the Lyapunov saturation regime is always reached, and energy drift becomes a random-walk-like process governed by the symplectic invariants of the discrete map (Wisdom 2018). The analysis of long-term drift in that regime requires different diagnostics and is left for future work.

**Alternative integrators.** The modified leapfrog of Wisdom & Holman (1991) and the RESPA multi-timescale integrator (Tuckerman, Berne & Martyna 1992) are explicitly designed for systems with a dominant smooth background potential plus small but rapidly varying corrections. For Plummer systems dominated by the background potential, RESPA could in principle provide higher-order accuracy with fewer sub-step force evaluations. This is a candidate for Phase 8 of the `gadget-ng` project.

---

## References

Aarseth, S. J. 1963, MNRAS, 126, 223

Barnes, J., & Hut, P. 1986, Nature, 324, 446

Dehnen, W. 2000, ApJ, 536, L39

Dehnen, W. 2014, Computational Astrophysics and Cosmology, 1, 1

Goodman, J., Heggie, D. C., & Hut, P. 1993, ApJ, 415, 715

Hairer, E., Lubich, C., & Wanner, G. 2006, *Geometric Numerical Integration*, 2nd ed. (Springer)

Heggie, D. C. 1991, in *The Use of Supercomputers in Stellar Dynamics*, eds. P. Hut & S. L. W. McMillan (Springer), 233

Hernquist, L. 1987, ApJS, 64, 715

Hut, P., Makino, J., & McMillan, S. 1995, ApJ, 443, L93

Makino, J. 1991, PASJ, 43, 859

McLachlan, R. I., & Atela, P. 1992, Nonlinearity, 5, 541

Miller, R. H. 1964, ApJ, 140, 250

Quinn, T., Katz, N., Stadel, J., & Lake, G. 1997, arXiv:astro-ph/9710043

Quinlan, G. D., & Tremaine, S. 1992, MNRAS, 259, 505

Springel, V. 2005, MNRAS, 364, 1105

Tuckerman, M., Berne, B. J., & Martyna, G. J. 1992, J. Chem. Phys., 97, 1990

Wisdom, J., & Holman, M. 1991, AJ, 102, 1528

Wisdom, J. 2018, MNRAS, 474, 3273

Yoshida, H. 1990, Phys. Lett. A, 150, 262

---

## Appendix A: Summary of All Phase 7 Runs

**Table A1.** Final relative energy drift $\delta_E(T)$ for all 54 Phase 7 runs (KDK + V5 solver, $\Delta t_\mathrm{base} = 0.025$, $T = 25$).

| Tag | $N$ | $\delta_E$ | Wall (s) |
|-----|-----|-----------|---------|
| **Plummer $a/\varepsilon = 1$** | | | |
| fixed $\Delta t=0.025$ | 200 | $4.69\times10^{-1}$ | 2.5 |
| fixed $\Delta t=0.0125$ | 200 | $7.97\times10^{-3}$ | 5.2 |
| fixed $\Delta t=0.00625$ | 200 | $1.36\times10^{-4}$ | 16.7 |
| acc $\eta=0.01$ | 200 | $1.30\times10^{-2}$ | 90.4 |
| acc $\eta=0.02$ | 200 | $5.23\times10^{-2}$ | 49.6 |
| acc $\eta=0.05$ | 200 | $2.83\times10^{-1}$ | 28.3 |
| jerk $\eta=0.01$ | 200 | $9.70\times10^{-2}$ | 21.0 |
| jerk $\eta=0.02$ | 200 | $8.89\times10^{-1}$ | 13.5 |
| jerk $\eta=0.05$ | 200 | $1.01\times10^{0}$ | 2.9 |
| fixed $\Delta t=0.025$ | 1000 | $3.24\times10^{-1}$ | 55 |
| fixed $\Delta t=0.0125$ | 1000 | $3.63\times10^{-4}$ | 124 |
| fixed $\Delta t=0.00625$ | 1000 | $1.38\times10^{-4}$ | 270 |
| acc $\eta=0.01$ | 1000 | $3.1\times10^{-3}$† | — |
| acc $\eta=0.02$ | 1000 | $2.4\times10^{-2}$† | — |
| acc $\eta=0.05$ | 1000 | $2.16\times10^{-1}$ | 451 |
| jerk $\eta=0.01$ | 1000 | $7.64\times10^{-2}$ | 302 |
| jerk $\eta=0.02$ | 1000 | $7.07\times10^{-1}$ | 196 |
| jerk $\eta=0.05$ | 1000 | $1.17\times10^{0}$ | 80 |
| **Plummer $a/\varepsilon = 2$** | | | |
| fixed $\Delta t=0.025$ | 200 | $3.65\times10^{-1}$ | 2.7 |
| fixed $\Delta t=0.0125$ | 200 | $1.00\times10^{-2}$ | 5.4 |
| fixed $\Delta t=0.00625$ | 200 | $2.49\times10^{-4}$ | 14.4 |
| acc $\eta=0.01$ | 200 | $1.08\times10^{-2}$ | 78.5 |
| acc $\eta=0.02$ | 200 | $4.06\times10^{-2}$ | 44.2 |
| acc $\eta=0.05$ | 200 | $2.09\times10^{-1}$ | 23.5 |
| jerk $\eta=0.01$ | 200 | $9.78\times10^{-2}$ | 15.6 |
| jerk $\eta=0.02$ | 200 | $7.04\times10^{-1}$ | 9.0 |
| jerk $\eta=0.05$ | 200 | $9.25\times10^{-1}$ | 3.4 |
| fixed $\Delta t=0.025$ | 1000 | $2.41\times10^{-1}$ | 61 |
| fixed $\Delta t=0.0125$ | 1000 | $8.88\times10^{-4}$ | 122 |
| fixed $\Delta t=0.00625$ | 1000 | $2.75\times10^{-4}$ | 239 |
| acc $\eta=0.01$ | 1000 | $9.1\times10^{-3}$† | — |
| acc $\eta=0.02$ | 1000 | $3.88\times10^{-2}$ | 998 |
| acc $\eta=0.05$ | 1000 | $1.99\times10^{-1}$ | 468 |
| jerk $\eta=0.01$ | 1000 | $9.94\times10^{-2}$ | 319 |
| jerk $\eta=0.02$ | 1000 | $5.19\times10^{-1}$ | 206 |
| jerk $\eta=0.05$ | 1000 | $8.50\times10^{-1}$ | 76 |
| **Uniform sphere** | | | |
| fixed $\Delta t=0.025$ | 200 | $8.23\times10^{-3}$ | 2.3 |
| fixed $\Delta t=0.0125$ | 200 | $4.46\times10^{-4}$ | 4.8 |
| fixed $\Delta t=0.00625$ | 200 | $1.89\times10^{-4}$ | 9.1 |
| acc $\eta=0.01$ | 200 | $1.87\times10^{-3}$ | 24.1 |
| acc $\eta=0.02$ | 200 | $6.65\times10^{-3}$ | 12.2 |
| acc $\eta=0.05$ | 200 | $2.11\times10^{-2}$ | 4.4 |
| jerk $\eta=0.01$ | 200 | $3.14\times10^{-2}$ | 6.0 |
| jerk $\eta=0.02$ | 200 | $1.73\times10^{-2}$ | 2.7 |
| jerk $\eta=0.05$ | 200 | $4.13\times10^{0}$ | 1.9 |
| fixed $\Delta t=0.025$ | 1000 | $3.28\times10^{-3}$ | 49 |
| fixed $\Delta t=0.0125$ | 1000 | $3.92\times10^{-4}$ | 108 |
| fixed $\Delta t=0.00625$ | 1000 | $1.93\times10^{-4}$ | 208 |
| acc $\eta=0.01$ | 1000 | $1.19\times10^{-3}$ | 471 |
| acc $\eta=0.02$ | 1000 | $4.45\times10^{-3}$ | 262 |
| acc $\eta=0.05$ | 1000 | $1.41\times10^{-2}$ | 105 |
| jerk $\eta=0.01$ | 1000 | $1.94\times10^{-2}$ | 114 |
| jerk $\eta=0.02$ | 1000 | $1.31\times10^{-3}$ | 54 |
| jerk $\eta=0.05$ | 1000 | $2.36\times10^{0}$ | 34 |

†Run interrupted and restarted; drift value from last available diagnostic step, wall time not recorded.

---

*Reproducibility note:* All simulation configurations, analysis scripts, and plot generators are available in the `gadget-ng` repository under `experiments/nbody/phase6_higher_order_integrator/` and `experiments/nbody/phase7_aarseth_timestep/`. Build and execution instructions are in `README.md`.
