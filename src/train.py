"""Training script for the non-v0 PEMFC PGNN pipeline.

This script trains a physics-guided regressor with:
- supervised data loss on scaled voltage,
- a derivative-based physics constraint on dV/dt,
- validation monitoring, LR scheduling, and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_loader import PEMFCDataset
from model import PhysicsGuidedNN


def train_pgml():
    """Train PGNN model and save both weights and scaler-aware checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on hardware: {device}")

    # Reproducibility for splits and parameter initialization.
    torch.manual_seed(42)
    
    # Load and preprocess dataset from canonical training CSV.
    dataset = PEMFCDataset("data/raw/synthetic_matrix.csv")

    # Build train/validation split for generalization monitoring.
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Model + optimizer + LR scheduler.
    model = PhysicsGuidedNN(input_dim=len(dataset.feature_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )
    # Robust regression loss for noisy synthetic measurements.
    data_criterion = nn.SmoothL1Loss(beta=0.5)
    
    epochs = 400
    physics_lambda = 0.005

    # Convert physical recovery-rate threshold (V/hour) into scaled-space units:
    # d(y_scaled)/d(t_scaled) = (std_t / std_y) * d(y)/d(t)
    time_idx = dataset.time_feature_index
    time_std = float(dataset.scaler_X.scale_[time_idx])
    voltage_std = float(dataset.scaler_y.scale_[0])
    max_allowable_recovery_v_per_h = 0.005
    max_allowable_recovery_scaled = (time_std / voltage_std) * max_allowable_recovery_v_per_h

    best_val_loss = float("inf")
    best_state = None
    patience = 50
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        # -------------------------------
        # Train phase
        # -------------------------------
        model.train()
        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Enable gradients on inputs for derivative-based physics penalty.
            X_batch.requires_grad_(True)
            
            optimizer.zero_grad()
            
            # Forward pass.
            V_pred = model(X_batch)
            
            # Supervised data term.
            loss_data = data_criterion(V_pred, y_batch)
            
            # Physics term: penalize unrealistically fast positive voltage recovery.
            dV_dX = torch.autograd.grad(
                outputs=V_pred, 
                inputs=X_batch, 
                grad_outputs=torch.ones_like(V_pred), 
                create_graph=True
            )[0]
            
            dV_dt = dV_dX[:, time_idx] 

            # ReLU keeps penalty active only when derivative exceeds threshold.
            loss_physics = torch.mean(torch.relu(dV_dt - max_allowable_recovery_scaled))
            
            # Total objective.
            loss_total = loss_data + (physics_lambda * loss_physics)
            
            loss_total.backward()
            # Gradient clipping helps stabilize occasional spikes.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total_loss += loss_total.item()
            epoch_data_loss += loss_data.item()
            epoch_physics_loss += loss_physics.item()

        # -------------------------------
        # Validation phase (data loss only)
        # -------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_hat = model(X_val)
                val_loss += data_criterion(y_hat, y_val).item()

        train_total_avg = epoch_total_loss / len(train_loader)
        train_data_avg = epoch_data_loss / len(train_loader)
        train_phys_avg = epoch_physics_loss / len(train_loader)
        val_avg = val_loss / len(val_loader)

        # Keep the best model checkpoint by validation loss.
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Reduce LR on validation plateau.
        scheduler.step(val_avg)

        if no_improve_epochs >= patience:
            print(f"[*] Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
            break
            
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"TrainTotal: {train_total_avg:.6f} | "
                f"TrainData: {train_data_avg:.6f} | "
                f"TrainPhys: {train_phys_avg:.6f} | "
                f"ValData: {val_avg:.6f} | "
                f"LR: {current_lr:.2e}"
            )

    # Restore the best-performing model before saving.
    if best_state is not None:
        model.load_state_dict(best_state)

    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pgnn_weights.pth")

    # Save a richer checkpoint (weights + scaler stats + feature metadata)
    # so evaluation can exactly reproduce training-time normalization.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_columns": dataset.feature_columns,
        "target_column": dataset.target_column,
        "x_mean": dataset.scaler_X.mean_.tolist(),
        "x_scale": dataset.scaler_X.scale_.tolist(),
        "y_mean": dataset.scaler_y.mean_.tolist(),
        "y_scale": dataset.scaler_y.scale_.tolist(),
    }
    torch.save(checkpoint, "models/pgnn_checkpoint.pth")

    print("[*] Model weights saved to models/pgnn_weights.pth")
    print("[*] Training checkpoint saved to models/pgnn_checkpoint.pth")

if __name__ == "__main__":
    train_pgml()