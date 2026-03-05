import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ['TiO2_Loading', 'RH_Percent', 'Time_Hours']
TARGET_COLUMN = 'Voltage'


def _canonical(text: str) -> str:
    return ''.join(ch for ch in text.lower() if ch.isalnum())


def normalize_pemfc_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CSV headers to the schema expected by the trainer.

    Required final columns:
    - TiO2_Loading
    - RH_Percent
    - Time_Hours
    - Voltage
    """
    canonical_to_actual = {_canonical(col): col for col in df.columns}

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

    rename_map = {}
    for target_name, candidates in aliases.items():
        if target_name in df.columns:
            continue

        matched_source = None
        for candidate in candidates:
            if candidate in canonical_to_actual:
                matched_source = canonical_to_actual[candidate]
                break

        if matched_source is not None:
            rename_map[matched_source] = target_name

    if rename_map:
        df = df.rename(columns=rename_map)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df

class PEMFCDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        df = normalize_pemfc_schema(df)
        
        # extract targets
        self.X_raw = df[FEATURE_COLUMNS].values
        self.y_raw = df[[TARGET_COLUMN]].values
        
        # standardization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X_scaled = self.scaler_X.fit_transform(self.X_raw)
        self.y_scaled = self.scaler_y.fit_transform(self.y_raw)
        
        # to tensors
        self.X = torch.tensor(self.X_scaled, dtype=torch.float32)
        self.y = torch.tensor(self.y_scaled, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]