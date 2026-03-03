import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PEMFCDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # extract targets
        self.X_raw = df[['TiO2_Loading', 'RH_Percent', 'Time_Hours']].values
        self.y_raw = df[['Voltage']].values
        
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