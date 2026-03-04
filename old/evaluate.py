import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model import PhysicsGuidedNN
from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, normalize_pemfc_schema

def evaluate_digital_twin():
    print("[*] Loading synthetic baseline and normalizers...")
    # Load data to recreate the exact scaling parameters
    df = pd.read_csv("data/raw/synthetic_matrix.csv")
    df = normalize_pemfc_schema(df)
    
    # We must use the exact same scalers from training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(df[FEATURE_COLUMNS].values)
    scaler_y.fit(df[[TARGET_COLUMN]].values)
    
    print("[*] Loading trained PGNN weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsGuidedNN(input_dim=3, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("models/pgnn_weights.pth", weights_only=True))
    model.eval() # Lock dropout and batchnorm layers for inference

    # Isolate a highly aggressive degradation case for visualization
    test_loading = 0.1
    test_rh = 80.0
    
    # Extract the raw noisy data for this specific condition
    mask = (df['TiO2_Loading'] == test_loading) & (df['RH_Percent'] == test_rh)
    df_test = df[mask].sort_values(by='Time_Hours')
    
    # Generate a perfectly smooth time vector for the AI to predict over
    time_vector = np.linspace(0, 500, 500)
    X_infer = np.column_stack((
        np.full_like(time_vector, test_loading),
        np.full_like(time_vector, test_rh),
        time_vector
    ))
    
    # Scale inputs, predict, and inverse-scale back to real physical Volts
    X_infer_scaled = scaler_X.transform(X_infer)
    X_tensor = torch.tensor(X_infer_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad(): # Disable autograd for faster inference
        y_pred_scaled = model(X_tensor).cpu().numpy()
        
    y_pred_physical = scaler_y.inverse_transform(y_pred_scaled)

    # --- Generate Publication-Ready Plot ---
    plt.figure(figsize=(10, 6))
    
    # Plot the raw, noisy "hardware" data
    plt.scatter(df_test['Time_Hours'], df_test['Voltage'], 
                alpha=0.3, color='red', s=10, label="Noisy Sensor Data (w/ Flooding)")
    
    # Plot the AI's physics-constrained understanding
    plt.plot(time_vector, y_pred_physical, 
             color='blue', linewidth=3, label="PGNN Physics-Constrained Prediction")
    
    plt.title(f"PEMFC Digital Twin Validation\nTiO2 Loading: {test_loading} | RH: {test_rh}%", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Hours)", fontsize=12)
    plt.ylabel("Cell Voltage (V)", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    import os
    os.makedirs("data/processed", exist_ok=True)
    plt.savefig("data/processed/pgnn_validation.png", dpi=300, bbox_inches='tight')
    print("[*] Validation plot saved to data/processed/pgnn_validation.png")

if __name__ == "__main__":
    evaluate_digital_twin()