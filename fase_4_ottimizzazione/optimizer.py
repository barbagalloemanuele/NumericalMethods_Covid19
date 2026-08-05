import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------
# Funzioni MOCK per dimostrare la logica di calibrazione
# Nel progetto completo, queste si collegherebbero alle Fasi 1, 2 e 3.
# ---------------------------------------------------------

def simulate_diffusion(D):
    """
    Simula l'esecuzione dell'equazione di diffusione PDE per un dato
    coefficiente di diffusione D. (Usa il CG della Fase 3 sotto il cofano).
    Qui ritorniamo una distribuzione "sintetica" di contagi in funzione di D.
    """
    # Creiamo una griglia 10x10 sintetica = 100 nodi
    np.random.seed(42)
    base_dist = np.random.rand(100) 
    
    # La quantità di contagi finale varia in modo non lineare in base a D
    return base_dist * np.exp(-D * 2.0)

def get_real_data():
    """
    Rappresenta l'estrazione dei dati della Protezione Civile (Fase 1).
    Immaginiamo che, nella realtà, il virus abbia avuto un D = 0.35.
    """
    return simulate_diffusion(D=0.35)


# ---------------------------------------------------------
# Cuore dell'Ottimizzazione: Minimi Quadrati
# ---------------------------------------------------------

def cost_function(D_array, real_data):
    """
    La funzione obiettivo/costo. Calcola i residui.
    Ritorna il vettore della differenza tra:
    Dati_Simulati(D) - Dati_Reali
    L'algoritmo di ottimizzazione modificherà D finché la somma dei 
    quadrati di questo vettore non sarà minima (zero).
    """
    D_guess = D_array[0]
    simulated_data = simulate_diffusion(D_guess)
    return simulated_data - real_data

if __name__ == "__main__":
    print("--- Calibrazione del Modello (Data Fitting & Minimi Quadrati) ---\n")
    print("Recupero dati storici dal DB (ground truth)...")
    real_data = get_real_data()
    
    # Partiamo con un'ipotesi "sbagliata" sul coefficiente D (es. 1.0)
    initial_guess = [1.0]
    
    print(f"Ipotesi (guess) iniziale per il coefficiente D: {initial_guess[0]}")
    print("Avvio ottimizzazione non lineare (Levenberg-Marquardt)...")
    
    # scipy.optimize.least_squares applica l'algoritmo LM o TRF
    # per trovare il parametro D che minimizza la cost_function
    res = least_squares(cost_function, initial_guess, args=(real_data,), method='lm')
    
    print("\nOttimizzazione completata!")
    print(f"Risultato ottimale (D_star) trovato: {res.x[0]:.4f}")
    print(f"Costo finale (somma dei quadrati degli scarti): {res.cost:.2e}")
    print(f"Numero di iterazioni (valutazioni funzione): {res.nfev}")
    
    if abs(res.x[0] - 0.35) < 1e-4:
        print("\nSUCCESSO! L'algoritmo ha individuato matematicamente il parametro corretto (0.35) basandosi solo sull'osservazione dei dati finali.")
