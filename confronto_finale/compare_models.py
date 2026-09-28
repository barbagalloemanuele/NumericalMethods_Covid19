import os
import matplotlib.pyplot as plt
import numpy as np

# Impostiamo uno stile moderno
plt.style.use('seaborn-v0_8-darkgrid')

def create_loss_landscape():
    """Genera un plot concettuale della convergenza del Parametro: Adam vs L-BFGS."""
    epoche_adam = np.linspace(1, 50000, 500)
    # L'errore di Adam scende all'inizio e poi stalla (linea piatta) a 0.0042 (0.3542 - 0.3500)
    errore_adam = 0.1 * np.exp(-epoche_adam / 4000) + 0.0042
    
    epoche_lbfgs = np.linspace(1, 7100, 200)
    # L-BFGS calibra perfettamente l'errore a zero (usiamo 1e-6 per plot in scala logaritmica)
    errore_lbfgs = 0.1 * np.exp(-epoche_lbfgs / 800) + 1e-6
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epoche_adam, errore_adam, label='PINN - Ottimizzatore Adam', color='#ff7f0e', linewidth=2)
    plt.semilogy(epoche_lbfgs, errore_lbfgs, label='PINN - Ottimizzatore L-BFGS', color='#2ca02c', linewidth=3)
    
    plt.title("Convergenza del Parametro Fisico (Adam vs L-BFGS)", fontsize=15, weight='bold')
    plt.xlabel("Iterazioni (Epoche / Valutazioni)", fontsize=12)
    plt.ylabel("Errore Assoluto |D_pred - 0.35| (Log Scale)", fontsize=12)
    plt.legend(fontsize=11)
    
    # Annotazioni
    plt.annotate('Stallo netto di Adam (Errore costante a 0.0042)\ndovuto alla Gradient Pathology', 
                 xy=(40000, 0.00425), xytext=(20000, 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
                 
    plt.annotate("L-BFGS azzera l'errore in 7100 iterazioni", 
                 xy=(7100, 1e-6), xytext=(15000, 1e-5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
                 
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(out_dir, "loss_landscape.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_mesh_free_scatter():
    """Genera lo scatter plot 2D per visualizzare la logica Mesh-Free delle PINN."""
    # Data points (500)
    x_data = np.random.rand(500)
    y_data = np.random.rand(500)
    
    # Physics points (2000)
    x_phys = np.random.rand(2000)
    y_phys = np.random.rand(2000)
    
    plt.figure(figsize=(8, 8))
    
    # Disegniamo prima i physics points (più chiari, in background)
    plt.scatter(x_phys, y_phys, color='#d62728', alpha=0.3, s=10, label='Physics Collocation Points (2000)')
    # Poi i data points (più scuri, in evidenza)
    plt.scatter(x_data, y_data, color='#1f77b4', alpha=0.9, s=25, edgecolor='black', label='Sensori Reali / Data Points (500)')
    
    plt.title("Rappresentazione Architettura Mesh-Free (PINN)", fontsize=15, weight='bold')
    plt.xlabel("Dominio Spaziale X", fontsize=12)
    plt.ylabel("Dominio Spaziale Y", fontsize=12)
    
    # Impostiamo i limiti del dominio quadrato
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Aggiungiamo un box informativo
    info_text = (
        "Le PINN non richiedono matrici sparse o griglie strutturate.\n"
        "L'Intelligenza Artificiale apprende la PDE da coordinate casuali,\n"
        "garantendo estrema flessibilità in assenza di dati (sensori guasti)."
    )
    plt.text(0.5, -0.05, info_text, ha='center', va='top', fontsize=11, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
             
    # La legenda fuori dal grafico
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=11)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(out_dir, "mesh_free_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_comparison():
    print("--- Avvio Fase 6: Confronto Numerico vs PINN ---")
    
    # 1. Grafico Avanzato (Convergenza Parametrica)
    create_loss_landscape()
    
    # 2. Grafici Didattici (Fisica e Spazio)
    create_mesh_free_scatter()
    
    # 2. Grafici Base (Accuratezza e Tempi)
    D_real = 0.3500
    D_numerico = 0.3500 
    tempo_numerico = 0.05 
    
    pinn_results = {}
    modelli_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "approccio_pinn", "fase_5_pinn", "modelli_addestrati")
    
    # Raccogliamo TUTTI i risultati presenti
    for env in ["local", "cluster"]:
        for opt in ["adam", "lbfgs"]:
            result_file = os.path.join(modelli_dir, env, f"result_{opt}.txt")
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    lines = f.readlines()
                    d_scoperto = float(lines[4].split(":")[1].strip())
                    tempo_training = float(lines[3].split(":")[1].replace("s", "").strip())
                    epoche = int(lines[2].split(":")[1].strip())
                    
                    label = f"PINN\n({env.capitalize()} - {opt.upper()})"
                    pinn_results[label] = {"D": d_scoperto, "tempo": tempo_training}
    
    labels = ["Numerico\n(L-M)"] + list(pinn_results.keys())
    D_vals = [D_numerico] + [res["D"] for res in pinn_results.values()]
    tempo_vals = [tempo_numerico] + [res["tempo"] for res in pinn_results.values()]
    
    # Generiamo una palette di colori dinamica
    colors = ['#1f77b4'] # Il primo è sempre blu (Numerico)
    colors += ['#d62728', '#ff7f0e', '#9467bd', '#8c564b'][:len(pinn_results)]
    
    # BAR CHART ACCURATEZZA
    # BAR CHART COMBINATO: ACCURATEZZA E TEMPI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Subplot 1: Accuratezza ---
    bars1 = ax1.bar(labels, D_vals, color=colors, edgecolor='black', linewidth=1.5)
    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax1.axhline(y=D_real, color='#2ca02c', linestyle='--', linewidth=2, label=f"Reale (Ground Truth): {D_real:.4f}")
    ax1.set_title("Scoperta del Parametro D", fontsize=14, weight='bold')
    ax1.set_ylabel("Valore del Parametro D", fontsize=12)
    ax1.set_ylim(0, max(max(D_vals), D_real) * 1.25)
    ax1.legend()
    
    # --- Subplot 2: Tempi Computazionali ---
    bars2 = ax2.bar(labels, tempo_vals, color=colors, edgecolor='black', linewidth=1.5)
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.05), f"{yval:.2f} s", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax2.set_title("Tempi Computazionali (Log Scale)", fontsize=14, weight='bold')
    ax2.set_ylabel("Tempo (Secondi)", fontsize=12)
    ax2.set_yscale('log')
    
    plt.suptitle("Risultati Finali: Accuratezza vs Costo Computazionale", fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(out_dir, "confronto_finale.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n--- Confronto Completato! Trovati {len(pinn_results)} risultati PINN. ---")

if __name__ == "__main__":
    run_comparison()
