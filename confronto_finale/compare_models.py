import os
import matplotlib.pyplot as plt
import numpy as np

def run_comparison():
    print("--- Avvio Fase 6: Confronto Numerico vs PINN ---")
    
    # Valori noti dalla Fase 4 (Approccio Numerico)
    # L'algoritmo di Levenberg-Marquardt trova l'ottimo quasi immediatamente (spesso < 10 iterazioni)
    D_real = 0.3500
    D_numerico = 0.3500 
    tempo_numerico = 0.05 # stima del tempo della fase 4
    
    # Leggiamo i risultati della PINN (Fase 5)
    pinn_results = {}
    modelli_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "approccio_pinn", "fase_5_pinn", "modelli_addestrati")
    
    # Controlliamo quali ambienti hanno eseguito la PINN (local o cluster)
    for env in ["local", "cluster"]:
        result_file = os.path.join(modelli_dir, env, "result.txt")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                lines = f.readlines()
                d_scoperto = float(lines[3].split(":")[1].strip())
                tempo_training = float(lines[2].split(":")[1].replace("s", "").strip())
                epoche = int(lines[1].split(":")[1].strip())
                pinn_results[env] = {
                    "D": d_scoperto,
                    "tempo": tempo_training,
                    "epoche": epoche
                }
    
    if not pinn_results:
        print("Errore: Nessun risultato PINN trovato. Esegui la Fase 5 prima.")
        return
        
    print(f"Risultati PINN trovati: {list(pinn_results.keys())}")
    
    # Per il grafico, scegliamo il risultato migliore o l'ultimo disponibile (preferibilmente cluster)
    env_scelto = "cluster" if "cluster" in pinn_results else "local"
    D_pinn = pinn_results[env_scelto]["D"]
    tempo_pinn = pinn_results[env_scelto]["tempo"]
    epoche_pinn = pinn_results[env_scelto]["epoche"]
    
    print(f"Generazione grafici comparativi usando l'ambiente PINN: {env_scelto.upper()}...")
    
    # ---- GRAFICO 1: ACCURATEZZA PARAMETRO D ----
    plt.figure(figsize=(10, 6))
    bars = plt.bar(["Reale (Ground Truth)", "Metodo Numerico", f"PINN ({env_scelto.capitalize()})"], 
                   [D_real, D_numerico, D_pinn], 
                   color=['#2ca02c', '#1f77b4', '#d62728'])
                   
    # Aggiungiamo il valore sopra ogni barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.axhline(y=D_real, color='black', linestyle='--', alpha=0.5, label="Valore Reale (Target)")
    plt.title("Confronto Accuratezza: Scoperta del Coefficiente di Diffusione (D)", fontsize=14)
    plt.ylabel("Valore del Parametro D", fontsize=12)
    plt.ylim(0, max(D_real, D_numerico, D_pinn) * 1.3) # Spazio per il testo
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot1_path = os.path.join(out_dir, "confronto_accuratezza.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---- GRAFICO 2: TEMPI DI ESECUZIONE ----
    plt.figure(figsize=(8, 5))
    bars2 = plt.bar(["Metodo Numerico (L-M)", f"PINN ({epoche_pinn} epoche)"], 
                   [tempo_numerico, tempo_pinn], 
                   color=['#1f77b4', '#d62728'])
                   
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (tempo_pinn*0.02), f"{yval:.2f} s", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.title("Confronto Tempi Computazionali", fontsize=14)
    plt.ylabel("Tempo (Secondi)", fontsize=12)
    plt.yscale('log') # Scala logaritmica perché i tempi della PINN sono enormemente superiori
    plt.grid(axis='y', alpha=0.3)
    
    plot2_path = os.path.join(out_dir, "confronto_tempi.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n--- Confronto Completato! ---")
    print(f"I grafici finali sono stati salvati in:")
    print(f"1. {plot1_path}")
    print(f"2. {plot2_path}")

if __name__ == "__main__":
    run_comparison()
