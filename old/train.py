import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import PEMFCDataset
from model import PhysicsGuidedNN

def train_pgml():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # verify if nvidia gpu is working
    print(f"[*] Training on hardware: {device}")
    
    dataset = PEMFCDataset("data/raw/synthetic_matrix.csv")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = PhysicsGuidedNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_criterion = nn.MSELoss()
    
    epochs = 1000
    physics_lambda = 0.1 # Weight of the physical constraint
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # track gradients on the input for dV/dt
            X_batch.requires_grad_(True) 
            
            optimizer.zero_grad()
            
            # forward pass, data fed
            V_pred = model(X_batch) 
            
            # data loss (MSE)
            loss_data = mse_criterion(V_pred, y_batch)
            
            # PHYSICS LOSS: Bounded Recovery Constraint
            dV_dX = torch.autograd.grad(
                outputs=V_pred, 
                inputs=X_batch, 
                grad_outputs=torch.ones_like(V_pred), 
                create_graph=True
            )[0]
            
            dV_dt = dV_dX[:, 2] 
            
            # FIX: Define the maximum physically allowable voltage recovery rate
            max_allowable_recovery = 0.05 
            
            #oOnly penalize the network if it predicts a recovery faster than the physical limit.
            # subtract the threshold. If dV_dt is 0.02 (normal), ReLU(0.02 - 0.05) = 0 (no penalty)
            # If dV_dt is 0.10 (impossible), ReLU(0.10 - 0.05) = 0.05 (penalty)
            loss_physics = torch.mean(torch.relu(dV_dt - max_allowable_recovery))
            
            # Total Loss
            loss_total = loss_data + (physics_lambda * loss_physics)
            
            loss_total.backward()
            optimizer.step()
            
            epoch_loss += loss_total.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss/len(dataloader):.6f}")

    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pgnn_weights.pth")
    print("[*] Model weights saved to models/pgnn_weights.pth")

if __name__ == "__main__":
    train_pgml()