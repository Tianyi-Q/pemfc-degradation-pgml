"""Training script for the PEMFC Physics-Guided ML pipeline.
(Pipeline step 3b — trains the model defined in model.py)

Core idea: Physics-Guided Machine Learning (PGML)
--------------------------------------------------
A standard neural network minimises only a *data loss* (how far its
predictions are from the labels).  That works, but the model is free to
learn patterns that violate physics — for instance, predicting that a
degrading fuel cell suddenly *recovers* voltage over time.

PGML adds a second *physics loss* term that encodes domain knowledge as a
soft constraint.  Here, the constraint is:

    "Voltage should not increase faster than 0.005 V per hour."

This is implemented by computing the partial derivative dV/dt with PyTorch
autograd and penalising any positive derivative that exceeds the threshold.
The total loss is:

    L_total = L_data  +  lambda * L_physics

where lambda (physics_lambda) controls the strength of the physics penalty.

Other training features:
- SmoothL1Loss (Huber loss) for robustness to the AR(1) noise in the data.
- Adam optimiser with weight decay for mild L2 regularisation.
- ReduceLROnPlateau scheduler: halves LR when validation loss stalls.
- Early stopping (patience 50): prevents overfitting and saves time.
- Checkpoint saving: stores model weights + scaler statistics so that
  evaluate.py can reproduce the exact same normalisation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_loader import PEMFCDataset
from model import PhysicsGuidedNN


def train_pgml():
    """Train the PGNN model and save weights + scaler-aware checkpoint.

    The function runs the full training loop:
    1. Load and scale the dataset.
    2. Split into 80 % train / 20 % validation.
    3. For each epoch, compute data loss + physics loss, backprop, step.
    4. Monitor validation loss for early stopping and LR scheduling.
    5. Save the best model to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on hardware: {device}")

    # Reproducibility: fix the random seed so that the train/val split
    # and weight initialisation are identical every run.
    torch.manual_seed(42)
    
    # Load the synthetic CSV and let PEMFCDataset handle schema
    # normalisation, feature selection, and StandardScaler fitting.
    dataset = PEMFCDataset("data/raw/synthetic_matrix.csv")

    # 80/20 train-validation split.  The validation set is NEVER used
    # for gradient updates — it exists only to monitor generalisation
    # and trigger early stopping / LR reduction.
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoaders handle batching and shuffling automatically.
    # Batch size 256 for training (shuffled each epoch for stochastic
    # gradient descent); 512 for validation (no shuffling needed).
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # ── Model, optimiser, and scheduler ──
    model = PhysicsGuidedNN(input_dim=len(dataset.feature_columns)).to(device)

    # Adam is the go-to optimiser for deep learning.  weight_decay adds
    # a small L2 penalty on parameters to prevent them from growing huge.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # ReduceLROnPlateau monitors a metric (here: validation loss) and
    # halves the learning rate (factor=0.5) whenever the metric has not
    # improved for `patience` epochs.  This lets training start fast and
    # then fine-tune with smaller steps.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # "min" because lower val loss is better
        factor=0.5,      # multiply LR by 0.5 on plateau
        patience=10,     # wait 10 epochs before reducing
        min_lr=1e-5,     # never go below this LR
    )

    # SmoothL1Loss (aka Huber loss) behaves like L2 (MSE) for small errors
    # and like L1 (MAE) for large errors.  This makes it robust to the
    # occasional outlier caused by AR(1) coloured noise in the data.
    # The beta parameter sets the transition point between L1 and L2.
    data_criterion = nn.SmoothL1Loss(beta=0.5)
    
    epochs = 400
    # physics_lambda controls how strongly the physics penalty influences
    # training relative to the data loss.  Too high → the model ignores
    # the data to satisfy the constraint; too low → the constraint is
    # ineffective.  0.005 was found through manual tuning.
    physics_lambda = 0.005

    # ── Convert the physical dV/dt threshold into scaled-space units ──
    # The model works in StandardScaler space, so a threshold expressed
    # in real V/hour must be converted:
    #   d(y_scaled)/d(t_scaled) = (std_time / std_voltage) * d(y)/d(t)
    # This ensures the physics penalty is applied correctly regardless
    # of the numeric scale of the features.
    time_idx = dataset.time_feature_index
    time_std = float(dataset.scaler_X.scale_[time_idx])
    voltage_std = float(dataset.scaler_y.scale_[0])
    max_allowable_recovery_v_per_h = 0.005   # real-world limit [V/h]
    max_allowable_recovery_scaled = (time_std / voltage_std) * max_allowable_recovery_v_per_h

    # Early-stopping bookkeeping.
    best_val_loss = float("inf")
    best_state = None
    patience = 50         # how many epochs without improvement before we stop
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        # -------------------------------
        # Train phase
        # -------------------------------
        # model.train() enables training-mode behaviour (dropout, batchnorm
        # running stats — not used here, but good practice).
        model.train()
        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # We need gradients on inputs (not just on weights) because
            # the physics loss requires d(output)/d(input_time).  By
            # default PyTorch only tracks gradients for parameters, so
            # we explicitly enable it on the input tensor.
            X_batch.requires_grad_(True)
            
            # Zero old gradients so they don't accumulate from previous batch.
            optimizer.zero_grad()
            
            # Forward pass: features in → predicted scaled voltage out.
            V_pred = model(X_batch)
            
            # ── Data loss ──
            # How far the prediction is from the true scaled voltage.
            loss_data = data_criterion(V_pred, y_batch)
            
            # ── Physics loss (dV/dt constraint) ──
            # Use autograd to compute the Jacobian of V_pred w.r.t. all
            # input features.  dV_dX has shape (batch, num_features).
            # We only care about the column corresponding to time.
            dV_dX = torch.autograd.grad(
                outputs=V_pred, 
                inputs=X_batch, 
                grad_outputs=torch.ones_like(V_pred),  # "seed" for chain rule
                create_graph=True   # keep the computation graph so we can
                                    # backprop through the physics loss too
            )[0]
            
            # Extract just the dV/dt column.
            dV_dt = dV_dX[:, time_idx] 

            # Penalise only positive derivatives that exceed the threshold.
            # ReLU(dV_dt - threshold) is zero when voltage is falling
            # (normal degradation) and positive when voltage is recovering
            # "too fast" (unphysical).  The mean aggregates over the batch.
            loss_physics = torch.mean(torch.relu(dV_dt - max_allowable_recovery_scaled))
            
            # ── Total loss ──
            # Weighted sum: data fidelity + physics regularisation.
            loss_total = loss_data + (physics_lambda * loss_physics)
            
            loss_total.backward()
            # Gradient clipping caps the maximum gradient norm to 1.0.
            # This prevents occasional exploding-gradient spikes (common
            # when the physics loss creates sharp curvature in the loss
            # landscape) from destabilising training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update weights.
            optimizer.step()
            
            epoch_total_loss += loss_total.item()
            epoch_data_loss += loss_data.item()
            epoch_physics_loss += loss_physics.item()

        # -------------------------------
        # Validation phase (data loss only)
        # -------------------------------
        # We evaluate on the held-out 20 % to check generalisation.
        # Physics loss is NOT included here because we only want to
        # know "how accurate is the model on unseen data".
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

        # Keep the best model snapshot (by validation loss).
        # We clone the state dict to CPU so it survives further training.
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Ask the scheduler to check whether validation loss improved.
        # If it hasn't for 10 consecutive epochs, LR is halved.
        scheduler.step(val_avg)

        # Early stopping: if validation hasn't improved for `patience`
        # epochs, further training is likely overfitting — stop now.
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

    # Restore the best-performing model (not the last epoch, which
    # may have started overfitting) before saving.
    if best_state is not None:
        model.load_state_dict(best_state)

    import os
    os.makedirs("models", exist_ok=True)

    # Save plain weights (lightweight, for quick loading).
    torch.save(model.state_dict(), "models/pgnn_weights.pth")

    # Save a rich checkpoint that bundles:
    #   - model weights
    #   - feature column names (so evaluate.py knows the expected input order)
    #   - StandardScaler mean/std for both X and y (so evaluate.py can
    #     reproduce the exact same normalisation without re-fitting)
    # This makes the checkpoint fully self-contained: you can ship it
    # to another machine and run evaluation without the original dataset.
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