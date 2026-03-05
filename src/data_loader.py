"""Dataset and schema utilities for PEMFC training/evaluation.
(Pipeline step 2 of 4)

This module sits between the raw CSV (produced by generate_matrix.py) and the
PyTorch training loop (train.py).  Its three jobs are:

1. **Schema normalisation** – real-world CSV files come with inconsistent
   headers ("RH_Percent" vs "Relative Humidity" vs "rh").  The
   `normalize_pemfc_schema` function maps any reasonable variant to the
   canonical column names the rest of the pipeline expects.

2. **Feature selection & scaling** – selects the core features (loading, RH,
   time) plus any available optional state features (current density,
   temperature, hydration, flooding, ECSA).  All features and the target
   (Voltage) are standardised to zero mean / unit variance via
   sklearn StandardScaler so the neural network trains stably.

3. **PyTorch Dataset** – wraps the scaled tensors in a class compatible with
   `torch.utils.data.DataLoader`, letting the training loop iterate over
   mini-batches automatically.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── Core feature set ────────────────────────────────────────────────────────
# These three columns MUST exist in any CSV passed to the pipeline.
# They represent the experimental conditions (loading, humidity) and the
# temporal axis (time).  Without them, neither training nor evaluation
# can proceed.
FEATURE_COLUMNS = ['TiO2_Loading', 'RH_Percent', 'Time_Hours']
TARGET_COLUMN = 'Voltage'

# Optional state-like features provide extra information the model can use
# if present.  They come from the physics simulation (generate_matrix.py)
# and would also be available in a real experiment with appropriate sensors.
# If any of these columns are missing from the CSV the pipeline carries on
# without them — the model simply receives fewer inputs.
OPTIONAL_FEATURE_COLUMNS = [
    'CurrentDensity_Acm2',   # operating current density [A cm^-2]
    'Temp_K',                # lumped cell temperature [K]
    'MembraneLambda',        # membrane water content proxy [3..22]
    'FloodingState',         # liquid-water accumulation [0..1]
    'ECSA_Norm',             # normalised catalyst surface area [0.5..1]
]


def _canonical(text: str) -> str:
    """Collapse a column header to a lowercase alphanumeric key.

    Examples:
        'RH_Percent'       -> 'rhpercent'
        'Relative Humidity' -> 'relativehumidity'
        'TiO2 (wt%)'       -> 'tio2wt'

    This lets us match headers that differ only in casing, underscores,
    spaces, or punctuation — a common problem when CSV files come from
    different labs or instruments.
    """
    return ''.join(ch for ch in text.lower() if ch.isalnum())


def normalize_pemfc_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map incoming CSV headers to canonical column names via fuzzy matching.

    For each required column (TiO2_Loading, RH_Percent, Time_Hours, Voltage)
    the function checks whether it already exists.  If not, it searches
    through a priority-ordered alias list (e.g. 'Relative Humidity' →
    'RH_Percent').  Matching is done on the _canonical() form so that
    differences in casing, underscores, or whitespace are ignored.

    Raises ValueError if any required column is still missing after
    alias resolution — this forces early, clear errors instead of cryptic
    KeyError deep inside training code.
    """
    # Build a lookup from canonical form -> original column name.
    canonical_to_actual = {_canonical(col): col for col in df.columns}

    # Alias table: for each canonical target, a list of plausible alternate
    # spellings ordered by priority (most specific first).  If somebody
    # hands us a CSV labelled "humidity" instead of "RH_Percent", we still
    # find it.
    aliases = {
        'TiO2_Loading': [
            'tio2loading', 'loading', 'catalystloading', 'tio2wt', 'tio2'
        ],
        'RH_Percent': [
            'rhpercent', 'rh', 'relativehumidity', 'humidity', 'rhpct'
        ],
        'Time_Hours': [
            'timehours', 'time', 'hours', 't', 'timeh'
        ],
        'Voltage': [
            'voltage', 'cellvoltage', 'voltagev', 'vcell', 'v'
        ],
    }

    # Attempt to rename any missing required columns using their aliases.
    rename_map = {}
    for target_name, candidates in aliases.items():
        if target_name in df.columns:
            continue  # Already present under the canonical name.

        matched_source = None
        for candidate in candidates:
            if candidate in canonical_to_actual:
                matched_source = canonical_to_actual[candidate]
                break  # First match wins (priority order).

        if matched_source is not None:
            rename_map[matched_source] = target_name

    if rename_map:
        df = df.rename(columns=rename_map)

    # Hard-fail if any required column could not be resolved.
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df

class PEMFCDataset(Dataset):
    """PyTorch dataset for PEMFC tabular regression.

    Wraps a single CSV file into scaled float32 tensors ready for
    DataLoader.  The scaling (zero-mean, unit-variance via StandardScaler)
    is critical: neural networks train much more reliably when all input
    features live on a similar numeric scale rather than mixing values
    like 0.05 (loading) with 333.0 (temperature in Kelvin).

    The fitted scaler statistics (mean, std) are stored as attributes so
    train.py can save them into the checkpoint and evaluate.py can
    reproduce the exact same normalisation at inference time.
    """

    def __init__(self, csv_file):
        """Load CSV, normalize schema, select features, and build scaled tensors.

        Steps:
        1. Read the raw CSV and run schema normalisation.
        2. Determine which optional columns are present.
        3. Fit StandardScaler on inputs and target independently.
        4. Store scaled numpy arrays then convert to float32 Tensors.
        """
        df = pd.read_csv(csv_file)
        df = normalize_pemfc_schema(df)

        # Dynamically include optional features that exist in this CSV.
        # This lets the same code work both on the full synthetic matrix
        # (which has all 8 columns) and on minimal real-data CSVs that
        # may only have loading, RH, time, and voltage.
        available_optional = [col for col in OPTIONAL_FEATURE_COLUMNS if col in df.columns]
        self.feature_columns = FEATURE_COLUMNS + available_optional
        self.target_column = TARGET_COLUMN

        # Record which position in the feature vector corresponds to time.
        # train.py needs this index to compute dV/dt for the physics penalty
        # (it takes the gradient of the model output w.r.t. this specific
        # input dimension).
        self.time_feature_index = self.feature_columns.index('Time_Hours')
        
        # Raw (unscaled) arrays kept for scaler fitting and diagnostics.
        self.X_raw = df[self.feature_columns].values
        self.y_raw = df[[self.target_column]].values
        
        # StandardScaler transforms each column to zero mean and unit
        # variance:  x_scaled = (x - mean) / std.
        # This is done separately for inputs and target so that the model
        # predicts in "scaled voltage space" and we invert the transform
        # after inference to get real volts back.
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X_scaled = self.scaler_X.fit_transform(self.X_raw)
        self.y_scaled = self.scaler_y.fit_transform(self.y_raw)
        
        # PyTorch tensors — float32 is the standard precision for GPU training.
        self.X = torch.tensor(self.X_scaled, dtype=torch.float32)
        self.y = torch.tensor(self.y_scaled, dtype=torch.float32)
        
    def __len__(self):
        """Return the number of data points (required by DataLoader)."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Return (features, target) for one data point (required by DataLoader)."""
        return self.X[idx], self.y[idx]