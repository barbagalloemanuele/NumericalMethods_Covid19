import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

# Impostiamo il seed per riproducibilità dei test
torch.manual_seed(42)
np.random.seed(42)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) per modellare la diffusione spaziale.
    """
    def __init__(self):
        super(PINN, self).__init__()
        # Input: (x, y, t) -> Output: u (contagi)
        # Usiamo un MLP (Multi-Layer Perceptron)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(), # Tanh è fondamentale perché le sue derivate seconde non sono nulle (a differenza di ReLU)
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # --- LA MAGIA DEL DATA FITTING ---
        # Trattiamo il parametro fisico 'D' (coefficiente di diffusione) come un peso della rete.
        # Lo inizializziamo a 1.0, e la backpropagation lo ottimizzerà basandosi sui dati.
        self.D = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, y, t):
        # Concateniamo i tensori lungo la dimensione 1 (colonne)
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)

def compute_physics_loss(model, x, y, t):
    """
    Calcola il residuo dell'equazione differenziale (Physics Loss).
    PDE: u_t - D*(u_xx + u_yy) = 0
    """
    # Richiediamo i gradienti per x, y, t per poter calcolare le derivate parziali
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, y, t)
    
    # 1. Derivata temporale: du/dt
    # create_graph=True è essenziale per poter calcolare derivate di ordine superiore
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # 2. Derivate spaziali del primo ordine: du/dx, du/dy
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # 3. Derivate spaziali del secondo ordine (Laplaciano): d²u/dx², d²u/dy²
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    # Calcolo del residuo (se la fisica è rispettata, residual == 0)
    residual = u_t - model.D * (u_xx + u_yy)
    
    # La Physics Loss è il Mean Squared Error (MSE) del residuo rispetto a 0
    loss_physics = torch.mean(residual**2)
    return loss_physics

def generate_mock_data(num_points=1000):
    """
    Simula l'estrazione di dati reali dalla Protezione Civile (ground truth).
    Generiamo punti sparsi e usiamo una funzione analitica fittizia 
    in cui il parametro D reale (nascosto) è 0.35.
    """
    # Generiamo coordinate casuali nel dominio [0, 1]
    x = torch.rand(num_points, 1)
    y = torch.rand(num_points, 1)
    t = torch.rand(num_points, 1)
    
    # Il parametro "segreto" della natura che la rete deve scoprire
    D_real = 0.35 
    
    # Soluzione analitica fittizia che rispetta parzialmente una dinamica di diffusione
    u = torch.exp(-D_real * t) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    
    return x, y, t, u

def train_pinn(epochs=1000, lr=1e-3, device='cpu', env='local'):
    print(f"--- Avvio Training PINN su {str(device).upper()} (Ambiente: {env}) ---")
    
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 1. Dati Storici (Data Points)
    # Simula i punti estratti dal CSV in Fase 1
    x_data, y_data, t_data, u_data = generate_mock_data(500)
    x_data, y_data, t_data, u_data = x_data.to(device), y_data.to(device), t_data.to(device), u_data.to(device)
    
    # 2. Punti di Collocazione (Physics Points)
    # Punti causali nel dominio spazio-temporale usati solo per far rispettare l'equazione differenziale
    x_phys, y_phys, t_phys, _ = generate_mock_data(2000)
    x_phys, y_phys, t_phys = x_phys.to(device), y_phys.to(device), t_phys.to(device)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # --- LOSS SUI DATI REALI ---
        u_pred = model(x_data, y_data, t_data)
        loss_data = torch.mean((u_pred - u_data)**2)
        
        # --- LOSS SULLA FISICA (PDE) ---
        loss_phys = compute_physics_loss(model, x_phys, y_phys, t_phys)
        
        # --- LOSS TOTALE MULTI-OBIETTIVO ---
        loss = loss_data + loss_phys
        
        loss.backward()
        optimizer.step()
        
        # Loggare il progresso
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss Totale: {loss.item():.4f} (Dati: {loss_data.item():.4f}, Fisica: {loss_phys.item():.4f}) | Parametro Scoperto D: {model.D.item():.4f}")
            
    end_time = time.time()
    train_time = end_time - start_time
    d_scoperto = model.D.item()
    
    print(f"\nTraining completato in {train_time:.2f} secondi.")
    print(f"==================================================")
    print(f"   Valore D scoperto dalla PINN: {d_scoperto:.4f}")
    print(f"   Valore D reale (Protezione Civile): 0.3500")
    print(f"==================================================")
    
    # Salvataggio del Modello in cartelle strutturate per ambiente
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelli_addestrati", env)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "pinn_model.pth")
    torch.save(model.state_dict(), save_path)
    
    result_path = os.path.join(save_dir, "result.txt")
    with open(result_path, "w") as f:
        f.write(f"Ambiente: {env}\n")
        f.write(f"Epoche: {epochs}\n")
        f.write(f"Tempo di training: {train_time:.2f} s\n")
        f.write(f"D_scoperto: {d_scoperto:.4f}\n")
        f.write(f"Loss_finale: {loss.item():.4e}\n")
        
    print(f"\nOutput salvati con successo in:\n-> {save_dir}")
    
    return model

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestramento PINN per diffusione spaziale.")
    parser.add_argument("--epochs", type=int, default=1000, help="Numero di epoche per l'addestramento.")
    parser.add_argument("--env", type=str, choices=["local", "cluster"], default="local", help="Ambiente di esecuzione per separare gli output.")
    args = parser.parse_args()

    # Rilevamento automatico per usare l'accelerazione hardware disponibile
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    train_pinn(epochs=args.epochs, lr=1e-3, device=device, env=args.env)
