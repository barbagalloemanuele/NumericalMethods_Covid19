import numpy as np
import scipy.sparse as sp

def build_laplacian_2d(nx, ny):
    """
    Costruisce la matrice sparsa per l'operatore Laplaciano 2D
    usando differenze finite (5-point stencil) su una griglia nx x ny.
    Ritorna la matrice in formato CSR (Compressed Sparse Row) per efficienza
    nei calcoli dei metodi iterativi.
    """
    N = nx * ny
    
    # Diagonale principale (-4)
    main_diag = -4.0 * np.ones(N)
    
    # Diagonali adiacenti sull'asse x (sinistra e destra)
    off_diag_x = np.ones(N - 1)
    # Rimuoviamo la connessione ai bordi (quando saltiamo riga nella matrice appiattita)
    off_diag_x[nx-1::nx] = 0.0
    
    # Diagonali adiacenti sull'asse y (sopra e sotto)
    off_diag_y = np.ones(N - nx)
    
    # Costruiamo la matrice usando le diagonali indicate (0 è la principale, -1/1 sono le adiacenti)
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, -1, 1, -nx, nx]
    
    # Usiamo sp.diags per costruire direttamente in formato sparso (evitando memory overflow)
    A = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
    
    return A

if __name__ == "__main__":
    # Dimensione della griglia scelta in Fase 1
    nx, ny = 50, 50
    
    print(f"Costruzione della matrice Laplaciana 2D per una griglia {nx}x{ny}...")
    A = build_laplacian_2d(nx, ny)
    
    print(f"Dimensioni del sistema lineare (matrice A): {A.shape}")
    print(f"Numero di elementi non nulli (nnz): {A.nnz}")
    
    # Verifichiamo la sparsità della matrice
    total_elements = A.shape[0] * A.shape[1]
    sparsity = 1.0 - (A.nnz / total_elements)
    print(f"Sparsità della matrice: {sparsity:.4%} (elementi nulli vs totali)")
    
    # Creiamo un termine noto fittizio b per mostrare il sistema Ax = b
    b = np.ones(A.shape[0])
    print(f"Dimensioni del vettore incognito x e del termine noto b: {b.shape}")
