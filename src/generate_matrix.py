"""
Compared to v0 I added physics-informed dynamics

Why make this imaginary data:
- Real degradation datasets are expensive and time-consuming to collect
- For model development and testing I need "data"

What is modeled (at a simplified but physically motivated level):
- Reversible voltage via Nernst relation
- Activation, ohmic, and concentration losses
- Dynamic internal states: membrane hydration, flooding, temperature, and ECSA decay
- Experiment matrix over TiO2 loading and relative humidity

I might have bitten more than I can chew here... but the previous version was clearly too simple.

What is not modeled explicitly:
- Full gas-channel transport PDEs
- Detailed catalyst-layer multi-phase transport
- Stack-level manifold/flow-field effects

Because I had worked mainly on GDLs, the catalysis and the inner workings of flow fields are put aside for  the moment. 

Use this as a calibration-ready baseline: tune coefficients against real polarization and
durability data before relying on absolute values that I just eyeballed here and there

I have yet to test all of this on actual .fcd results, but that's on my to-do list, meaning that I will get to it...probably:)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
F = 96485.3329         # C/mol Faraday constant
R = 8.314462618        # J/(mol*K) universal gas constant


def nernst_voltage(T, p_h2=1.0, p_o2=0.21, p_h2o=0.03):
    """
    Return simplified PEMFC reversible (open-circuit) voltage [V].

    This function captures the two key dependencies:
    1) Baseline temperature correction of standard potential.
    2) Partial-pressure correction via the Nernst logarithmic term.

    Notes:
    - Inputs are treated as effective partial pressures (dimensionless, relative scale).
    - Very small-value guards are included to avoid log(0).
    """
    e0 = 1.229 - 0.85e-3 * (T - 298.15)
    nernst = (R * T / (2 * F)) * np.log(max((p_h2 * np.sqrt(p_o2)) / max(p_h2o, 1e-6), 1e-9))
    return e0 + nernst


def plot_voltage_matrix(df, output_plot="data/raw/synthetic_matrix_plot.png"):
    """
    Plot Voltage vs Time for each TiO2 loading, with one curve per RH level.

    Layout:
    - 2x2 subplots (one panel per loading value in the current matrix).
    - Shared axes to make visual comparison across loadings easier.
    - Figure-level legend to avoid duplicate legends on each panel.
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
    seed=42
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
    # These shape activation, ohmic, and concentration loss magnitudes.
    alpha = 0.5
    i0_ref = 1e-6
    i_lim0 = 1.5
    B_conc = 0.03
    sigma0 = 10.0
    l_mem = 0.012
    A_cell = 25.0

    # Loading-dependent performance priors to match observed ranking:
    # 0.10 (best) > 0.05 > 0.20 > 0.30 (worst)
    loading_activity_factor = {0.05: 0.96, 0.10: 1.00, 0.20: 0.90, 0.30: 0.82}
    loading_contact_resistance = {0.05: 0.0035, 0.10: 0.0025, 0.20: 0.0045, 0.30: 0.0065}
    loading_flood_factor = {0.05: 1.00, 0.10: 0.95, 0.20: 1.06, 0.30: 1.18}
    loading_voltage_bias = {0.05: 0.008, 0.10: 0.015, 0.20: 0.001, 0.30: -0.010}

    # -------------------------------------------------------------------------
    # 3) Dynamic-state coefficients (tunable)
    # -------------------------------------------------------------------------
    # These govern temporal evolution of hydration, flooding, thermal response,
    # and catalyst degradation speed.
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
                # Current density has:
                # - a loading-dependent baseline,
                # - a mild diurnal-like sinusoidal modulation,
                # - small Gaussian perturbation.
                # Use near-constant demand profile; catalyst loading influences voltage
                # through kinetics/transport terms rather than large current differences.
                i_base = 0.58 + 0.15 * (loading - 0.10)
                i_t = i_base * (1.0 + 0.04 * np.sin(2 * np.pi * t / 24.0)) + rng.normal(0, 0.01)
                i_t = float(np.clip(i_t, 0.05, 1.2))

                # -----------------------------------------------------------------
                # 4b) State updates (discrete-time, Euler integration)
                # -----------------------------------------------------------------
                # Membrane hydration dynamics:
                # - relax toward RH-dependent target,
                # - increase with water production (current-linked),
                # - decrease via drying term.
                lam_target = 14.0 * rh_frac
                dlam = k_hum * (lam_target - lam) + k_prod * i_t - k_dry * (1 - rh_frac) * lam
                lam = float(np.clip(lam + dt * dlam, 3.0, 22.0))

                # Flooding dynamics:
                # - growth promoted by high RH and high current,
                # - reduced by drainage.
                flood_mult = loading_flood_factor.get(round(loading, 2), 1.0)
                dtheta = k_flood * flood_mult * max(rh_frac - 0.88, 0.0) * (i_t / i_lim0) - k_drain * theta_f
                theta_f = float(np.clip(theta_f + dt * dtheta, 0.0, 0.95))

                # Stochastic purge event: occasionally removes accumulated liquid water.
                if rng.random() < (purge_prob + 0.01 * max(rh_frac - 0.7, 0.0)) and rh >= 80:
                    theta_f *= 0.5

                # Lumped thermal balance:
                # - heat generation scales with i^2,
                # - cooling relaxes toward nominal operating temperature.
                dT = k_heat * (i_t ** 2) - k_cool * (T - 333.15)
                T = float(np.clip(T + dt * dT, 303.15, 353.15))

                # ECSA decay accelerates with flooding and elevated temperature.
                accel_T = np.exp((T - 333.15) / 40.0)
                ecsa *= np.exp(-k_deg * dt * (1 + 2 * theta_f) * accel_T)
                ecsa = float(np.clip(ecsa, 0.5, 1.0))

                # -----------------------------------------------------------------
                # 4c) Voltage model decomposition
                # -----------------------------------------------------------------
                # Total voltage is reversible voltage minus overpotentials:
                # V = E_nernst - eta_activation - eta_ohmic - eta_concentration

                # Water-vapor proxy used in Nernst correction.
                p_h2o = 0.01 + 0.07 * rh_frac + 0.05 * theta_f
                E = nernst_voltage(T, p_h2=1.0, p_o2=0.21, p_h2o=p_h2o)

                # Activation loss via Tafel-like term with temperature-adjusted i0.
                activity_factor = loading_activity_factor.get(round(loading, 2), 0.9)
                # Higher RH improves proton transport and effective kinetics.
                rh_kinetics_boost = 0.80 + 0.45 * rh_frac
                i0_eff = i0_ref * ecsa * activity_factor * rh_kinetics_boost * np.exp((T - 333.15) / 25.0)
                eta_act = (R * T / (alpha * F)) * np.log(max(i_t / max(i0_eff, 1e-12), 1.000001))

                # Ohmic loss from membrane ionic resistance.
                # Conductivity increases with hydration and generally with temperature.
                sigma_mem = sigma0 * (lam / 14.0) * np.exp(1268.0 * (1 / 303.15 - 1 / T))
                sigma_mem = max(sigma_mem, 0.2)
                r_mem = l_mem / (sigma_mem * A_cell)
                # Add small loading-specific contact resistance contribution.
                r_contact = loading_contact_resistance.get(round(loading, 2), 0.004)
                eta_ohm = i_t * A_cell * r_mem + i_t * r_contact

                # Concentration loss grows rapidly as current approaches limiting current,
                # and limiting current decreases under flooding.
                i_lim_eff = i_lim0 * ((1 - theta_f) ** 1.5)
                i_lim_eff = max(i_lim_eff, 0.2)
                frac = min(i_t / i_lim_eff, 0.98)
                eta_conc = -B_conc * np.log(1 - frac)

                # Net RH benefit term to ensure physically expected trend under this
                # simplified model (higher RH generally improves voltage).
                rh_voltage_gain = 0.020 * (rh_frac - 0.30)
                bias_loading = loading_voltage_bias.get(round(loading, 2), 0.0)

                v_model = E - eta_act - eta_ohm - eta_conc + rh_voltage_gain + bias_loading

                # -----------------------------------------------------------------
                # 4d) Noise model and record assembly
                # -----------------------------------------------------------------
                # AR(1)-style colored noise better mimics sensor/process correlation
                # than pure white noise.
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