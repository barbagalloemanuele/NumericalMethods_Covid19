import sys
import os
import time
import numpy as np
import scipy.sparse.linalg as spla

# Aggiungiamo la root del progetto al path per importare la funzione della fase 2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fase_2_pde.pde_discretization import build_laplacian_2d

def solve_system(A, b, method='cg', tol=1e-6):
    """
    Risolve il sistema lineare Ax = b usando metodi iterativi.
    Traccia il numero di iterazioni e il tempo di calcolo.
    """
    residuals = []
    
    def callback_cg(xk):
        residuals.append(1) # Traccia solo il numero di iterazioni

    def callback_gmres(pr_norm):
        residuals.append(pr_norm) # Nelle vecchie versioni riceve la norma del residuo
        
    print(f"--- Risoluzione tramite {method.upper()} ---")
    start_time = time.time()
    
    if method == 'cg':
        x, info = spla.cg(A, b, rtol=tol, atol=0, callback=callback_cg)
    elif method == 'gmres':
        # callback_type='pr_norm' è supportato in scipy per gmres
        x, info = spla.gmres(A, b, rtol=tol, atol=0, callback=callback_gmres, callback_type='legacy')
    else:
        raise ValueError("Metodo non supportato.")
        
    end_time = time.time()
    
    if info == 0:
        print(f"Convergenza raggiunta in {len(residuals)} iterazioni.")
    else:
        print(f"Convergenza NON raggiunta (info={info}).")
        
    print(f"Tempo di calcolo: {end_time - start_time:.4f} secondi.")
    
    # Calcolo residuo finale reale: ||Ax - b|| / ||b||
    final_residual = np.linalg.norm(b - A.dot(x)) / np.linalg.norm(b)
    print(f"Residuo relativo finale: {final_residual:.2e}\n")
    
    return x, residuals

if __name__ == "__main__":
    nx, ny = 50, 50
    # Costruiamo la matrice A (Laplaciano). 
    # Per il Conjugate Gradient la matrice deve essere Simmetrica e Definita Positiva (SPD).
    # Il nostro Laplaciano ha -4 sulla diagonale (è definito negativo). Lo moltiplichiamo per -1 per renderlo SPD.
    A = -build_laplacian_2d(nx, ny)
    N = A.shape[0]
    
    # Creiamo un termine noto b "random" ma controllato per riprodurre un caso di contagi
    np.random.seed(42)
    b = np.random.rand(N)
    
    print(f"Risoluzione del sistema di dimensione {N}x{N}...\n")
    
    # 1. Gradiente Coniugato (CG) - Ottimale per matrici SPD
    x_cg, res_cg = solve_system(A, b, method='cg', tol=1e-6)
    
    # 2. GMRES - Metodo generale (usato come benchmark di confronto)
    x_gmres, res_gmres = solve_system(A, b, method='gmres', tol=1e-6)

    # Confronto tra i due metodi numerici
    diff = np.linalg.norm(x_cg - x_gmres)
    print(f"Differenza assoluta tra le soluzioni dei due metodi: {diff:.2e}")
    if diff < 1e-5:
        print("I due metodi hanno prodotto risultati identici entro la tolleranza.")
