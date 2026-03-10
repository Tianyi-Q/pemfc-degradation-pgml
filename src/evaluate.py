"""Evaluation script for the PEMFC Physics-Guided ML pipeline.
(Pipeline step 4 of 4 — runs after train.py)

This script answers the question: "How well does the trained model
predict fuel-cell voltage across all operating conditions?"

Workflow:
1. Load the same synthetic CSV used for training.
2. Load the saved model checkpoint (weights + scaler statistics).
3. For every operating segment (loading x RH combination), run
   inference and compute per-segment and global error metrics.
4. Generate three publication-ready plots:
   a) Multi-segment time-series overlay (measured vs predicted).
   b) Parity plot (predicted vs measured scatter + ideal y = x line).
   c) Local comparison (single segment with zoomed early-time panel).
5. Save a CSV summary of per-segment metrics.

All outputs land in data/processed/ for downstream reporting.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from model import PhysicsGuidedNN
from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, normalize_pemfc_schema


def _metrics(y_true, y_pred):
    """Compute three standard regression metrics for a prediction vector.

    Returns (MAE, RMSE, R^2) where:
    - MAE  (Mean Absolute Error): average absolute deviation in volts.
           Easy to interpret — "on average, predictions are X V off".
    - RMSE (Root Mean Squared Error): like MAE but penalises large errors
           more heavily because squaring amplifies outliers.
    - R^2  (Coefficient of Determination): fraction of total variance
           explained by the model.  1.0 = perfect, 0.0 = no better than
           predicting the mean, <0 = worse than the mean.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return mae, rmse, r2


def evaluate_digital_twin():
    """Run the full evaluation workflow and save visual / CSV artifacts.

    This function ties together data loading, model inference, metric
    computation, and plot generation.  It operates in evaluation-only
    mode (model.eval(), torch.no_grad()) so no gradients are computed
    and memory usage stays low.
    """
    print("[*] Loading synthetic baseline and normalizers...")
    # Load data and normalize schema to expected canonical column names.
    df = pd.read_csv("data/raw/synthetic_matrix.csv")
    df = normalize_pemfc_schema(df)

    # Load the checkpoint saved by train.py.  It contains model weights
    # plus the StandardScaler statistics (mean, std) that were fitted on
    # the training data.  Using these ensures we normalise evaluation
    # inputs in exactly the same way as during training — any mismatch
    # would silently corrupt predictions.
    ckpt_path = "models/pgnn_checkpoint.pth"

    # Prefer checkpoint scalers; fall back to re-computing from the CSV
    # only when running with legacy weight files that lack scaler info.
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
        x_scale = np.asarray(checkpoint["x_scale"], dtype=np.float64)
        y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float64)
        y_scale = np.asarray(checkpoint["y_scale"], dtype=np.float64)
        feature_columns = checkpoint.get("feature_columns", FEATURE_COLUMNS)
        print("[*] Using saved training checkpoint scalers.")
    else:
        # Backward-compatible fallback if legacy weights exist without checkpoint.
        x_mean = df[FEATURE_COLUMNS].values.mean(axis=0)
        x_scale = df[FEATURE_COLUMNS].values.std(axis=0, ddof=0)
        y_mean = df[[TARGET_COLUMN]].values.mean(axis=0)
        y_scale = df[[TARGET_COLUMN]].values.std(axis=0, ddof=0)
        feature_columns = FEATURE_COLUMNS
        print("[!] Checkpoint not found, using fallback scalers from current CSV.")

    # Safety: if any feature has zero variance (constant column), set
    # its scale to 1.0 to avoid division by zero during normalisation.
    x_scale = np.where(x_scale == 0.0, 1.0, x_scale)
    y_scale = np.where(y_scale == 0.0, 1.0, y_scale)
    
    print("[*] Loading trained PGNN weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsGuidedNN(input_dim=len(feature_columns), hidden_dim=64).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(torch.load("models/pgnn_weights.pth", weights_only=True))
    # Switch to evaluation mode: disables dropout / batchnorm training
    # behaviour (not used in this architecture, but good practice).
    model.eval()

    # Ensure all expected feature columns are present in the current CSV.
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"CSV is missing checkpoint feature columns: {missing_features}. "
            f"Found columns: {list(df.columns)}"
        )

    loadings = sorted(df['TiO2_Loading'].unique())
    rhs = sorted(df['RH_Percent'].unique())

    # Enumerate every unique (loading, RH) operating segment and run
    # the trained model on each one.  This lets us compute per-segment
    # metrics and later visualise each condition separately.
    segment_frames = []
    metrics_rows = []

    # torch.no_grad() disables autograd tracking, saving memory and time
    # because we do not need gradients during evaluation.
    with torch.no_grad():
        for loading in loadings:
            for rh in rhs:
                seg = df[(df['TiO2_Loading'] == loading) & (df['RH_Percent'] == rh)].copy()
                seg = seg.sort_values(by='Time_Hours')
                if seg.empty:
                    continue

                # Normalise inputs using the training-time scaler stats,
                # run the model, then invert the target scaling to get
                # predictions back in real volts.
                X_infer = seg[feature_columns].values
                X_infer_scaled = (X_infer - x_mean) / x_scale
                X_tensor = torch.tensor(X_infer_scaled, dtype=torch.float32).to(device)

                # Model predicts in scaled-voltage space; undo:
                #   V_real = V_scaled * std_voltage + mean_voltage
                y_pred_scaled = model(X_tensor).cpu().numpy()
                y_pred = ((y_pred_scaled * y_scale) + y_mean).reshape(-1)

                seg['PredictedVoltage'] = y_pred
                segment_frames.append(seg)

                mae, rmse, r2 = _metrics(seg['Voltage'].values, y_pred)
                metrics_rows.append({
                    'TiO2_Loading': loading,
                    'RH_Percent': rh,
                    'MAE_V': mae,
                    'RMSE_V': rmse,
                    'R2': r2,
                    'NumPoints': len(seg),
                })

    # Concatenate all segments into one big DataFrame and build a
    # summary table of per-segment metrics.
    if not segment_frames:
        raise ValueError("No segments were available for evaluation.")

    pred_df = pd.concat(segment_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows).sort_values(['TiO2_Loading', 'RH_Percent'])

    # Global metrics across all points.
    global_mae, global_rmse, global_r2 = _metrics(pred_df['Voltage'].values, pred_df['PredictedVoltage'].values)
    print(
        f"[*] Global metrics | MAE: {global_mae:.5f} V | RMSE: {global_rmse:.5f} V | R2: {global_r2:.4f}"
    )
    print("[*] Per-segment metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # ------------------------------------------------------------------
    # Multi-segment validation plot
    # ------------------------------------------------------------------
    # One subplot per TiO2 loading level, arranged in a 2x2 grid.
    # Within each subplot, every RH condition is drawn as a pair of lines:
    #   - faded thin line  = measured (noisy sensor) voltage
    #   - vivid thick line = PGNN prediction
    # Using the same colour for both makes it easy to see how closely
    # the prediction tracks the measurement for each RH.
    n_loadings = len(loadings)
    # Two columns layout (2x2 for four loadings).
    ncols = 2
    nrows = int(np.ceil(n_loadings / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols).flatten()
    color_map = plt.cm.tab10(np.linspace(0, 1, len(rhs)))

    for i, loading in enumerate(loadings):
        ax = axes[i]
        sub = pred_df[pred_df['TiO2_Loading'] == loading]
        loading_rmse = metrics_df[metrics_df['TiO2_Loading'] == loading]['RMSE_V'].mean()

        for j, rh in enumerate(rhs):
            seg = sub[sub['RH_Percent'] == rh]
            if seg.empty:
                continue

            c = color_map[j]
            # Measured/noisy sensor trace (faded for contrast).
            ax.plot(
                seg['Time_Hours'],
                seg['Voltage'],
                linewidth=1.1,
                alpha=0.32,
                color=c,
            )
            # PGNN prediction trace (thicker/vivid for comparison).
            ax.plot(
                seg['Time_Hours'],
                seg['PredictedVoltage'],
                linewidth=2.4,
                alpha=0.98,
                color=c,
                label=f"RH {rh}%",
            )

        ax.set_title(f"TiO2 Loading = {loading:.2f} | Mean RMSE = {loading_rmse:.4f} V")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Cell Voltage (V)")
        t_min = float(sub['Time_Hours'].min()) if not sub.empty else 0.0
        t_max = float(sub['Time_Hours'].max()) if not sub.empty else 500.0
        ax.set_xlim(t_min, t_max)
        ax.set_xticks(np.arange(int(t_min), int(t_max) + 1, 50))
        ax.grid(True, linestyle='--', alpha=0.35)

    # Hide unused axes when number of loadings is odd.
    for k in range(len(loadings), len(axes)):
        axes[k].axis('off')

    # Shared legend and figure-level annotations.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "PEMFC Multi-Segment Validation",
        fontsize=15,
        fontweight='bold',
        y=0.99,
    )
    fig.text(
        0.5,
        0.955,
        "All traces are solid: faded lines = measured sensor data, vivid thicker lines = PGNN prediction",
        ha='center',
        va='center',
        fontsize=10,
    )
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(len(rhs), 4),
        frameon=False,
        title='Relative Humidity',
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.84])

    # ------------------------------------------------------------------
    # Parity plot: predicted vs measured voltage
    # ------------------------------------------------------------------
    # A parity plot is a scatter of (measured, predicted) for every data
    # point.  If the model were perfect, all dots would sit on the
    # diagonal y = x line.  Spread around that line shows prediction
    # error.  This is a standard way to visualise regression quality.
    fig_parity, ax_parity = plt.subplots(figsize=(7, 7))
    ax_parity.scatter(
        pred_df['Voltage'],
        pred_df['PredictedVoltage'],
        s=10,
        alpha=0.25,
        color='tab:blue',
    )
    v_min = float(min(pred_df['Voltage'].min(), pred_df['PredictedVoltage'].min()))
    v_max = float(max(pred_df['Voltage'].max(), pred_df['PredictedVoltage'].max()))
    ax_parity.plot([v_min, v_max], [v_min, v_max], 'k--', linewidth=1.5, label='Ideal y = x')
    ax_parity.set_title(
        f"Predicted vs Measured Voltage\nGlobal RMSE = {global_rmse:.4f} V | R2 = {global_r2:.4f}",
        fontsize=12,
        fontweight='bold',
    )
    ax_parity.set_xlabel("Measured Cell Voltage (V)")
    ax_parity.set_ylabel("Predicted Cell Voltage (V)")
    ax_parity.grid(True, linestyle='--', alpha=0.35)
    ax_parity.legend(loc='upper left')
    fig_parity.tight_layout()

    # Persist artifacts for downstream reporting.
    os.makedirs("data/processed", exist_ok=True)
    multi_plot_path = "data/processed/pgnn_validation_multisegment.png"
    parity_plot_path = "data/processed/pgnn_validation_parity.png"
    metrics_path = "data/processed/pgnn_validation_metrics.csv"

    fig.savefig(multi_plot_path, dpi=300, bbox_inches='tight')
    fig_parity.savefig(parity_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig_parity)

    metrics_df.to_csv(metrics_path, index=False)

    print(f"[*] Multi-segment validation plot saved to {multi_plot_path}")
    print(f"[*] Parity plot saved to {parity_plot_path}")
    print(f"[*] Segment metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # Local comparison: one loading at one temperature level
    # ------------------------------------------------------------------
    # This is a "close-up" view of a single operating regime.  It helps
    # you see fine-grained prediction behaviour (small ripples, noise
    # tracking) that gets lost in the multi-segment overview.
    # Two panels: full time horizon + early-time zoom for extra detail.
    #
    # We prefer loading = 0.10 (the "best" condition) when available.
    preferred_loading = 0.10
    local_loading = preferred_loading if preferred_loading in loadings else float(loadings[0])
    local_sub = pred_df[pred_df['TiO2_Loading'] == local_loading].copy()

    if not local_sub.empty:
        if 'Temp_K' in local_sub.columns:
            # Choose a representative temperature band around the median.
            # We start with a tight tolerance (+/- 0.6 K) and widen it
            # until we have at least 30 data points for a meaningful plot.
            temp_target = float(local_sub['Temp_K'].median())
            tolerance_candidates = [0.6, 1.2, 2.0]
            local_band = pd.DataFrame()

            for tol in tolerance_candidates:
                candidate = local_sub[np.abs(local_sub['Temp_K'] - temp_target) <= tol]
                if len(candidate) >= 30:
                    local_band = candidate
                    temp_tol = tol
                    break

            if local_band.empty:
                # Fallback: select nearest points by temperature distance.
                local_sub = local_sub.assign(_temp_dist=np.abs(local_sub['Temp_K'] - temp_target))
                local_band = local_sub.sort_values('_temp_dist').head(200).drop(columns=['_temp_dist'])
                temp_tol = float(local_band['Temp_K'].sub(temp_target).abs().max())

            local_title = (
                f"Local Comparison | Loading={local_loading:.2f}, "
                f"Temp~{temp_target:.2f} +/- {temp_tol:.2f} K"
            )
        else:
            # If temperature is unavailable, use one RH segment as local slice.
            local_rh = float(sorted(local_sub['RH_Percent'].unique())[0])
            local_band = local_sub[local_sub['RH_Percent'] == local_rh]
            local_title = f"Local Comparison | Loading={local_loading:.2f}, RH={local_rh:.0f}%"

        local_band = local_band.sort_values('Time_Hours')
        local_mae, local_rmse, local_r2 = _metrics(
            local_band['Voltage'].values,
            local_band['PredictedVoltage'].values,
        )

        print(
            f"[*] Local metrics | {local_title} | "
            f"MAE: {local_mae:.5f} V | RMSE: {local_rmse:.5f} V | R2: {local_r2:.4f} | "
            f"Points: {len(local_band)}"
        )

        # Two-panel local view: full horizon + early-time zoom for clarity.
        fig_local, axes_local = plt.subplots(2, 1, figsize=(11, 7), sharey=True)
        ax_local_full, ax_local_zoom = axes_local

        ax_local_full.plot(
            local_band['Time_Hours'],
            local_band['Voltage'],
            color='tab:gray',
            linewidth=1.8,
            alpha=0.7,
            label='Measured Performance',
        )
        ax_local_full.plot(
            local_band['Time_Hours'],
            local_band['PredictedVoltage'],
            color='tab:blue',
            linewidth=2.4,
            alpha=0.95,
            label='Predicted Performance',
        )
        ax_local_full.set_title(
            f"{local_title}\nRMSE={local_rmse:.4f} V, R2={local_r2:.4f}",
            fontsize=12,
            fontweight='bold',
        )
        ax_local_full.set_xlabel('Time (hours)')
        ax_local_full.set_ylabel('Cell Voltage (V)')
        ax_local_full.grid(True, linestyle='--', alpha=0.35)
        ax_local_full.legend(loc='best')

        zoom_window_hours = 60.0
        t0 = float(local_band['Time_Hours'].min())
        t1 = float(local_band['Time_Hours'].max())
        zoom_end = min(t0 + zoom_window_hours, t1)
        local_zoom = local_band[(local_band['Time_Hours'] >= t0) & (local_band['Time_Hours'] <= zoom_end)]
        if local_zoom.empty:
            local_zoom = local_band

        ax_local_zoom.plot(
            local_zoom['Time_Hours'],
            local_zoom['Voltage'],
            color='tab:gray',
            linewidth=1.8,
            alpha=0.7,
        )
        ax_local_zoom.plot(
            local_zoom['Time_Hours'],
            local_zoom['PredictedVoltage'],
            color='tab:blue',
            linewidth=2.4,
            alpha=0.95,
        )
        ax_local_zoom.set_title(f"Zoomed View (first {int(zoom_end - t0)} hours)", fontsize=11)
        ax_local_zoom.set_xlabel('Time (hours)')
        ax_local_zoom.set_ylabel('Cell Voltage (V)')
        ax_local_zoom.grid(True, linestyle='--', alpha=0.35)
        fig_local.tight_layout()

        local_plot_path = "data/processed/pgnn_validation_local_comparison.png"
        fig_local.savefig(local_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_local)
        print(f"[*] Local comparison plot saved to {local_plot_path}")

if __name__ == "__main__":
    evaluate_digital_twin()