"""
PEMFC Data Generator  (Pipeline step 1 of 4)
======================================================
This script is the FIRST stage of the pipeline.  It creates a synthetic but
physically motivated dataset that mimics what a real PEMFC degradation test
would produce.  The output CSV feeds directly into data_loader.py -> train.py
-> evaluate.py.

Compared to v0 I added physics-informed dynamics.

Why synthetic data?
- Real degradation datasets require hundreds of hours of cell operation and
  are expensive / time-consuming to collect.
- For model development and testing we need "data" that reproduces the right
  qualitative trends so the ML model learns physically plausible mappings.
- Once the pipeline is validated on synthetic data the same code can be
  pointed at real .fcd / .csv measurements with minimal changes.

Physics that IS modeled (simplified but motivated):
- Thermodynamic reversible voltage via Nernst equation.
- Three classical overpotential losses:
    * Activation (Tafel kinetics)  - energy barrier at the catalyst surface.
    * Ohmic (membrane ionic resistance) - proton-transport losses.
    * Concentration (mass-transport) - gas starvation near limiting current.
- Five dynamic internal states evolved with Euler integration:
    1. Membrane hydration (lambda)  - water content of the proton-exchange
       membrane; higher hydration -> lower ohmic resistance.
    2. Flooding state (theta_f) - liquid water accumulation in porous layers;
       excessive flooding blocks gas pathways and reduces limiting current.
    3. Cell temperature (T) - lumped thermal balance; heat from i^2 losses
       vs. cooling toward ambient.
    4. ECSA decay - normalised electrochemically-active surface area that
       decays irreversibly, simulating catalyst dissolution / Ostwald
       ripening over hundreds of hours.
    5. AR(1) coloured noise - correlated measurement noise that mimics
       real sensor drift better than pure white noise.
- A small experiment matrix over TiO2 catalyst loading and relative humidity
  (4 loadings x 4 RH levels = 16 operating segments of 500 h each).

What is NOT modeled explicitly:
- Full gas-channel transport PDEs.
- Detailed catalyst-layer multi-phase transport.
- Stack-level manifold / flow-field effects.

Because I had worked mainly on GDLs, the catalysis and the inner workings
of flow fields are put aside for the moment.

Calibration note:
All coefficients are hand-tuned placeholders.  Tune them against real
polarisation and durability data before trusting absolute voltage values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Physical constants ────────────────────────────────────────────────────
# These appear in every electrochemistry textbook and are used in the
# Nernst equation and Tafel kinetics below.
F = 96485.3329         # Faraday constant [C mol^-1] — charge per mole of electrons
R = 8.314462618        # Universal gas constant [J mol^-1 K^-1]


def nernst_voltage(T, p_h2=1.0, p_o2=0.21, p_h2o=0.03):
    """
    Compute the thermodynamic (open-circuit) reversible voltage of a
    hydrogen-oxygen fuel cell using the Nernst equation.

    This is the MAXIMUM voltage the cell could ever produce — before any
    losses are subtracted.  It depends on:
        T     – cell temperature [K]
        p_h2  – hydrogen partial pressure on the fuel side [atm]
        p_o2  – oxygen partial pressure on the air side [atm]
        p_h2o – water-vapour partial pressure [atm]

    Returns the reversible voltage E [V].
    """
    # Standard reversible potential at 25 °C is 1.229 V.
    # The linear correction (-0.85 mV/K) captures the thermodynamic fact
    # that higher temperature slightly lowers the equilibrium voltage
    # (entropy term of the Gibbs free-energy change).
    e0 = 1.229 - 0.85e-3 * (T - 298.15)

    # The logarithmic Nernst correction adjusts for actual gas pressures:
    #   - More H2 and O2 on the reactant side → higher voltage
    #   - More H2O product → lower voltage (Le Chatelier's principle)
    # The factor 2F arises because the H2/O2 reaction transfers 2 electrons.
    nernst = (R * T / (2 * F)) * np.log(max((p_h2 * np.sqrt(p_o2)) / max(p_h2o, 1e-6), 1e-9))

    return e0 + nernst


def plot_voltage_matrix(df, output_plot="data/raw/synthetic_matrix_plot.png"):
    """
    Plot Voltage vs Time for each TiO2 loading, with one curve per RH level:
    - 2x2 subplots (one panel per loading value)
    - Shared axes
    """
    loadings = sorted(df["TiO2_Loading"].unique())
    rhs = sorted(df["RH_Percent"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    # Each subplot corresponds to one loading condition.
    for i, loading in enumerate(loadings):
        ax = axes[i]
        sub = df[df["TiO2_Loading"] == loading]

        # Draw one voltage trajectory per RH setting within this loading.
        for rh in rhs:
            d = sub[sub["RH_Percent"] == rh]
            ax.plot(d["Time_Hours"], d["Voltage"], linewidth=1.1, label=f"RH {rh}%")

        ax.set_title(f"TiO2 Loading = {loading:.2f} (fraction)")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Cell Voltage (V)")
        ax.grid(alpha=0.3)

    # Use handles from the first axis because labels are consistent across panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Relative Humidity",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Synthetic PEMFC Degradation Matrix", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.90])

    # Ensure output directory exists before saving.
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig.savefig(output_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_synthetic_pemfc_data(
    output_path="data/raw/synthetic_matrix.csv",
    plot_path="data/raw/synthetic_matrix_plot.png",
    seed=42 #the answer to the ultimate question of life, the universe, and everything
):
    """
    Generate synthetic PEMFC time-series data over a small condition matrix.

    Output columns:
    - TiO2_Loading: catalyst/additive loading level (experiment factor)
    - RH_Percent: relative humidity condition
    - Time_Hours: simulation time
    - CurrentDensity_Acm2: operating current density
    - Temp_K: lumped cell temperature state
    - MembraneLambda: membrane water-content proxy
    - FloodingState: lumped liquid-water accumulation state [0..1]
    - ECSA_Norm: normalized catalyst activity proxy [0.5..1]
    - Voltage: terminal voltage after losses + colored noise

    The model is intentionally compact; coefficients are placeholders for calibration.
    """
    # Reproducible random-number generator for consistent synthetic datasets.
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # 1) Experiment matrix definition
    # -------------------------------------------------------------------------
    # These are the discrete operating points we iterate through.
    loadings = [0.05, 0.1, 0.2, 0.3]
    rhs = [30, 60, 80, 100]
    hours = np.arange(0, 500, 1.0)  # 1 h step
    dt = 1.0

    # -------------------------------------------------------------------------
    # 2) Static / electrochemical model parameters (tunable)
    # -------------------------------------------------------------------------
    # These constants shape how large each voltage-loss term is.  Every one
    # of them is a tuneable "knob" you would calibrate against real data.
    #
    #   alpha    – Charge-transfer coefficient (symmetry factor).  Controls
    #              the steepness of the Tafel slope.  0.5 is a common
    #              textbook default for a symmetric reaction barrier.
    #   i0_ref   – Exchange current density [A cm^-2].  This is the
    #              "background" reaction rate at equilibrium — very small
    #              for oxygen reduction (~1e-6), meaning the reaction is
    #              sluggish and needs a significant overpotential to run.
    #   i_lim0   – Limiting current density [A cm^-2].  The maximum
    #              current that can flow before gas starvation occurs.
    #   B_conc   – Concentration-loss coefficient [V].  Governs how
    #              sharply voltage drops as current approaches i_lim.
    #   sigma0   – Reference membrane ionic conductivity [S cm^-1].
    #              How easily protons hop through the membrane.
    #   l_mem    – Membrane thickness [cm].  Thinner membrane → lower
    #              ohmic resistance but weaker mechanical integrity.
    #   A_cell   – Active cell area [cm^2].  Scales total resistance.
    alpha = 0.5
    i0_ref = 1e-6
    i_lim0 = 1.5
    B_conc = 0.03
    sigma0 = 10.0
    l_mem = 0.012
    A_cell = 25.0

    # Loading-dependent performance priors.
    # In real PEMFC experiments, TiO2 additive loading has a non-monotonic
    # effect: a moderate amount (0.10) helps water management and catalyst
    # activity, but too much (0.30) clogs pores and increases contact
    # resistance.  These lookup tables encode that ranking:
    #   0.10 (best) > 0.05 > 0.20 > 0.30 (worst)
    #
    # activity_factor     – multiplier on exchange current density (higher = faster kinetics)
    # contact_resistance  – extra electronic resistance at interfaces [ohm]
    # flood_factor        – susceptibility to liquid-water flooding (>1 = floods more easily)
    # voltage_bias        – small constant offset to fine-tune absolute voltage level [V]
    loading_activity_factor = {0.05: 0.96, 0.10: 1.00, 0.20: 0.90, 0.30: 0.82}
    loading_contact_resistance = {0.05: 0.0035, 0.10: 0.0025, 0.20: 0.0045, 0.30: 0.0065}
    loading_flood_factor = {0.05: 1.00, 0.10: 0.95, 0.20: 1.06, 0.30: 1.18}
    loading_voltage_bias = {0.05: 0.008, 0.10: 0.015, 0.20: 0.001, 0.30: -0.010}

    # -------------------------------------------------------------------------
    # 3) Dynamic-state coefficients (tunable)
    # -------------------------------------------------------------------------
    # Each k_* controls how fast a particular internal state evolves per
    # time-step.  Think of them as "speed dials":
    #
    #   k_hum   – rate at which membrane hydration relaxes toward the
    #             RH-dependent equilibrium (higher = faster equilibration).
    #   k_prod  – water production rate from the electrochemical reaction
    #             (current generates water at the cathode → wets membrane).
    #   k_dry   – drying rate when ambient RH is low (membrane loses water).
    #   k_flood – rate of liquid-water accumulation in porous layers
    #             (only kicks in above ~88 % RH).
    #   k_drain – rate at which liquid water drains away naturally.
    #   k_heat  – heat generation coefficient from i² losses.
    #   k_cool  – heat removal rate (convective / conductive cooling toward
    #             the nominal 60 °C operating temperature).
    #   k_deg   – ECSA decay rate constant.  Small because catalyst
    #             degradation is a slow, irreversible process (hundreds of h).
    #   purge_prob – probability per time-step of a stochastic purge event
    #             that suddenly clears accumulated liquid water (models
    #             real-world gas-channel purging or breakthrough events).
    k_hum = 0.08
    k_prod = 0.10
    k_dry = 0.03
    k_flood = 0.05
    k_drain = 0.04
    k_heat = 0.8
    k_cool = 0.06
    k_deg = 2e-4
    purge_prob = 0.015

    data = []

    # -------------------------------------------------------------------------
    # 4) Nested simulation loop: condition matrix -> time stepping
    # -------------------------------------------------------------------------
    for loading in loadings:
        for rh in rhs:
            rh_frac = rh / 100.0

            # Initial states at t=0 for this operating condition.
            # They are intentionally simple proxies, not full multi-physics states.
            T = 333.15
            lam = 6.0 + 8.0 * rh_frac
            theta_f = max(0.0, (rh_frac - 0.7) * 0.2)
            ecsa = 1.0
            ar_noise = 0.0

            for t in hours:
                # -----------------------------------------------------------------
                # 4a) Operating profile generation
                # -----------------------------------------------------------------
                # Each hour we decide "how hard the cell is being driven".
                # The current density i_t has three components:
                #   1. A loading-dependent baseline (~0.58 A/cm^2) — different
                #      loadings shift the operating point slightly.
                #   2. A mild 24-hour sinusoidal cycle (+/- 4 %) that mimics
                #      diurnal demand fluctuation.
                #   3. A tiny Gaussian jitter (sigma 0.01) for realism.
                # The profile is intentionally near-constant so that
                # performance differences between conditions come from the
                # voltage model (kinetics, transport) rather than from
                # dramatically different load levels.
                i_base = 0.58 + 0.15 * (loading - 0.10)
                i_t = i_base * (1.0 + 0.04 * np.sin(2 * np.pi * t / 24.0)) + rng.normal(0, 0.01)
                i_t = float(np.clip(i_t, 0.05, 1.2))

                # -----------------------------------------------------------------
                # 4b) State updates (discrete-time, Euler integration)
                # -----------------------------------------------------------------
                # All five internal states are updated with simple first-order
                # ODEs solved by explicit Euler: x_new = x_old + dt * dx/dt.
                # This is the cheapest integrator; adequate for 1-hour steps
                # in a synthetic dataset.
                #
                # ── Membrane hydration (lambda) ──
                # Lambda tracks how much water is inside the Nafion membrane.
                # Three competing drivers:
                #   1. Relaxation toward the RH equilibrium (k_hum term) —
                #      membrane absorbs / desorbs water to match ambient RH.
                #   2. Water production from the electrochemical reaction
                #      (k_prod * i_t) — every proton drags water molecules.
                #   3. Drying at low RH (k_dry term) — dry gas strips water.
                # Lambda is clamped to [3, 22] (physical range for Nafion).
                lam_target = 14.0 * rh_frac
                dlam = k_hum * (lam_target - lam) + k_prod * i_t - k_dry * (1 - rh_frac) * lam
                lam = float(np.clip(lam + dt * dlam, 3.0, 22.0))

                # ── Flooding state (theta_f) ──
                # Liquid water can accumulate in the gas-diffusion layer and
                # catalyst pores.  Flooding only becomes significant when RH
                # exceeds ~88 % (the max() gate below).  Higher current also
                # promotes flooding because more water is produced.
                # Drainage opposes accumulation.  theta_f is clamped to [0, 0.95].
                flood_mult = loading_flood_factor.get(round(loading, 2), 1.0)
                dtheta = k_flood * flood_mult * max(rh_frac - 0.88, 0.0) * (i_t / i_lim0) - k_drain * theta_f
                theta_f = float(np.clip(theta_f + dt * dtheta, 0.0, 0.95))

                # Stochastic purge: in real systems, accumulated liquid water
                # occasionally breaks through or is purged.  We model this as
                # a random event that halves the flooding state.  It only
                # triggers at RH >= 80 % and becomes more likely at higher RH.
                if rng.random() < (purge_prob + 0.01 * max(rh_frac - 0.7, 0.0)) and rh >= 80:
                    theta_f *= 0.5

                # ── Lumped thermal balance ──
                # Heat is generated proportionally to i^2 (resistive / irreversible
                # losses) and removed by cooling toward the nominal 60 °C (333.15 K).
                # Clamped to [30 °C, 80 °C].
                dT = k_heat * (i_t ** 2) - k_cool * (T - 333.15)
                T = float(np.clip(T + dt * dT, 303.15, 353.15))

                # ── ECSA (catalyst surface area) decay ──
                # This is an IRREVERSIBLE degradation mechanism: the catalyst
                # slowly dissolves / agglomerates over hundreds of hours.
                # Decay is exponential and is ACCELERATED by:
                #   - elevated temperature (Arrhenius-like exp term), and
                #   - flooding (liquid water promotes Pt dissolution).
                # ECSA_Norm is clamped to [0.5, 1.0] — the model assumes
                # the cell never loses more than half its catalyst.
                accel_T = np.exp((T - 333.15) / 40.0)
                ecsa *= np.exp(-k_deg * dt * (1 + 2 * theta_f) * accel_T)
                ecsa = float(np.clip(ecsa, 0.5, 1.0))

                # -----------------------------------------------------------------
                # 4c) Voltage model decomposition
                # -----------------------------------------------------------------
                # The terminal cell voltage follows the classic decomposition:
                #
                #   V = E_nernst  (max thermodynamic voltage)
                #       - eta_act  (activation / kinetic loss)
                #       - eta_ohm  (ohmic / membrane-resistance loss)
                #       - eta_conc (concentration / mass-transport loss)
                #       + small corrections (RH benefit, loading bias)
                #
                # Each loss term chips away at the ideal Nernst voltage;
                # their sum determines the actual operating voltage.

                # Water-vapour partial pressure used in the Nernst correction.
                # It rises with humidity and with flooding (more liquid water
                # means more vapour at the electrode surface).
                p_h2o = 0.01 + 0.07 * rh_frac + 0.05 * theta_f
                E = nernst_voltage(T, p_h2=1.0, p_o2=0.21, p_h2o=p_h2o)

                # ── Activation loss (Tafel equation) ──
                # This is the energy penalty for making the sluggish oxygen-
                # reduction reaction happen at a finite rate.  The effective
                # exchange current density i0_eff depends on:
                #   - remaining catalyst area (ecsa),
                #   - loading activity multiplier,
                #   - RH boost (wetter membranes improve proton access to
                #     the catalyst), and
                #   - Arrhenius temperature dependence.
                # The Tafel logarithm means activation loss rises steeply
                # at low current then flattens at higher current.
                activity_factor = loading_activity_factor.get(round(loading, 2), 0.9)
                rh_kinetics_boost = 0.80 + 0.45 * rh_frac
                i0_eff = i0_ref * ecsa * activity_factor * rh_kinetics_boost * np.exp((T - 333.15) / 25.0)
                eta_act = (R * T / (alpha * F)) * np.log(max(i_t / max(i0_eff, 1e-12), 1.000001))

                # ── Ohmic loss (membrane + contact resistance) ──
                # Protons must cross the membrane, and their conductivity
                # sigma_mem depends on:
                #   - hydration level lambda (wetter = more conductive), and
                #   - temperature (Arrhenius term with 1268 K activation energy).
                # Resistance = thickness / (conductivity * area).  Contact
                # resistance adds a fixed electrode-interface penalty that
                # varies with loading.
                sigma_mem = sigma0 * (lam / 14.0) * np.exp(1268.0 * (1 / 303.15 - 1 / T))
                sigma_mem = max(sigma_mem, 0.2)
                r_mem = l_mem / (sigma_mem * A_cell)
                r_contact = loading_contact_resistance.get(round(loading, 2), 0.004)
                eta_ohm = i_t * A_cell * r_mem + i_t * r_contact

                # ── Concentration loss (mass-transport limitation) ──
                # As current approaches the limiting current, reactant gas
                # is consumed faster than it can diffuse to the catalyst.
                # The -ln(1 - i/i_lim) term captures the sharp voltage
                # cliff near starvation.  Flooding LOWERS the effective
                # limiting current because liquid water blocks gas pores.
                i_lim_eff = i_lim0 * ((1 - theta_f) ** 1.5)
                i_lim_eff = max(i_lim_eff, 0.2)
                frac = min(i_t / i_lim_eff, 0.98)
                eta_conc = -B_conc * np.log(1 - frac)

                # Small correction terms:
                #   rh_voltage_gain – enforces the physical expectation that
                #     higher RH generally improves performance (better proton
                #     transport, lower membrane resistance).
                #   bias_loading – fine-tunes absolute voltage per loading to
                #     match the desired ranking.
                rh_voltage_gain = 0.020 * (rh_frac - 0.30)
                bias_loading = loading_voltage_bias.get(round(loading, 2), 0.0)

                # ── Final model voltage (deterministic part) ──
                v_model = E - eta_act - eta_ohm - eta_conc + rh_voltage_gain + bias_loading

                # -----------------------------------------------------------------
                # 4d) Noise model and record assembly
                # -----------------------------------------------------------------
                # Real voltage sensors exhibit temporally correlated noise
                # (successive readings are not fully independent).  An AR(1)
                # process (autoregressive order 1) models this:
                #   noise_t = 0.7 * noise_{t-1} + white_noise
                # The coefficient 0.7 controls correlation strength; the
                # white-noise sigma (0.0015 V) sets the overall noise floor.
                # Final voltage is clamped to [0.35, 1.05] V for physical
                # plausibility (a real PEMFC cannot go below ~0.3 V or
                # above ~1.05 V under load).
                ar_noise = 0.7 * ar_noise + rng.normal(0, 0.0015)
                v_final = float(np.clip(v_model + ar_noise, 0.35, 1.05))

                data.append([
                    loading, rh, t, i_t, T, lam, theta_f, ecsa, v_final
                ])

    # -------------------------------------------------------------------------
    # 5) Build output table and save artifacts
    # -------------------------------------------------------------------------
    df = pd.DataFrame(
        data,
        columns=[
            "TiO2_Loading", "RH_Percent", "Time_Hours", "CurrentDensity_Acm2",
            "Temp_K", "MembraneLambda", "FloodingState", "ECSA_Norm", "Voltage"
        ],
    )

    # Persist both tabular data (CSV) and quick-look visualization (PNG).
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    plot_voltage_matrix(df, plot_path)

    print(f"[*] Synthetic matrix saved to {output_path} with {len(df)} records.")
    print(f"[*] Plot saved to {plot_path}")


if __name__ == "__main__":
    generate_synthetic_pemfc_data()