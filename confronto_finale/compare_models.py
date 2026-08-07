import os
import matplotlib.pyplot as plt
import numpy as np

# Impostiamo uno stile moderno
plt.style.use('seaborn-v0_8-darkgrid')

def create_radar_chart():
    """Genera un grafico Radar per un confronto qualitativo avanzato."""
    labels = np.array([
        'Velocità di Calcolo', 
        'Precisione Parametri (Dati Ideali)', 
        'Robustezza (Dati Mancanti/Sparsi)', 
        'Flessibilità (Mesh-Free)', 
        'Estrapolazione Continua (Tempo)'
    ])
    
    num_vars = len(labels)
    
    # Valori qualitativi da 0 a 10
    # Numerico è veloce, preciso su dati ideali, ma pessimo su dati mancanti/mesh
    val_numerico = [10, 10, 2, 0, 1] 
    # PINN è lenta, fatica sui parametri precisi, ma domina su dati sparsi e continui
    val_pinn = [2, 4, 9, 10, 10]
    
    # Angoli per il radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Chiudiamo il poligono
    val_numerico += val_numerico[:1]
    val_pinn += val_pinn[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot Numerico
    ax.plot(angles, val_numerico, color='#1f77b4', linewidth=2, label='Metodo Numerico (Matrici Sparse)')
    ax.fill(angles, val_numerico, color='#1f77b4', alpha=0.25)
    
    # Plot PINN
    ax.plot(angles, val_pinn, color='#d62728', linewidth=2, linestyle='solid', label='PINN (Deep Learning)')
    ax.fill(angles, val_pinn, color='#d62728', alpha=0.25)
    
    # Etichette e stile
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11, weight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([]) # nasconde i cerchi numerici interni per pulizia estetica
    
    plt.title('Radar Chart: Compromessi Architetturali', size=15, color='black', y=1.1, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(out_dir, "robustness_radar.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_loss_landscape():
    """Genera un plot concettuale della competizione delle Loss nelle PINN."""
    epoche = np.linspace(1, 50000, 500)
    
    # Simuliamo il decadimento tipico (Data loss scende subito, Physics loss crea interferenza)
    data_loss = 0.5 * np.exp(-epoche / 5000) + 0.005 * np.sin(epoche/1000)**2
    physics_loss = 0.8 * np.exp(-epoche / 15000) + 0.02 * np.cos(epoche/800)**2
    total_loss = data_loss + physics_loss
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epoche, total_loss, label='Total Loss', color='black', linewidth=3)
    plt.semilogy(epoche, data_loss, label='Data Loss (Aderenza dati)', color='#2ca02c', linewidth=2, linestyle='--')
    plt.semilogy(epoche, physics_loss, label='Physics Loss (Rispetto PDE)', color='#ff7f0e', linewidth=2, linestyle='-.')
    
    plt.title("Evoluzione Loss PINN: Conflitto dei Gradienti (Minimo Locale)", fontsize=15, weight='bold')
    plt.xlabel("Epoche di Addestramento", fontsize=12)
    plt.ylabel("Loss (Scala Logaritmica)", fontsize=12)
    plt.legend(fontsize=11)
    
    # Aggiungiamo un'annotazione
    plt.annotate('Stallo dei gradienti in\nminimi locali sub-ottimali', 
                 xy=(30000, 0.03), xytext=(35000, 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
                 
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(out_dir, "loss_landscape.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_comparison():
    print("--- Avvio Fase 6: Confronto Numerico vs PINN ---")
    
    # 1. Grafici Avanzati (Radar e Loss)
    create_radar_chart()
    create_loss_landscape()
    
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
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, D_vals, color=colors, edgecolor='black', linewidth=1.5)
                   
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.axhline(y=D_real, color='#2ca02c', linestyle='--', linewidth=2, label=f"Reale (Ground Truth): {D_real:.4f}")
    plt.title("Accuratezza: Scoperta del Coefficiente di Diffusione (D)", fontsize=15, weight='bold')
    plt.ylabel("Valore del Parametro D", fontsize=12)
    plt.ylim(0, max(max(D_vals), D_real) * 1.3)
    plt.legend()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(out_dir, "confronto_accuratezza.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # BAR CHART TEMPI
    plt.figure(figsize=(12, 5))
    bars2 = plt.bar(labels, tempo_vals, color=colors, edgecolor='black', linewidth=1.5)
                   
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.05), f"{yval:.2f} s", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.title("Confronto Tempi Computazionali (Log Scale)", fontsize=15, weight='bold')
    plt.ylabel("Tempo (Secondi)", fontsize=12)
    plt.yscale('log')
    
    plt.savefig(os.path.join(out_dir, "confronto_tempi.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n--- Confronto Completato! Trovati {len(pinn_results)} risultati PINN. ---")

if __name__ == "__main__":
    run_comparison()
