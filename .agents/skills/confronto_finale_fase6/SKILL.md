---
name: Fase 6 - Confronto Finale (Risultati e Deliverables)
description: Istruzioni per l'analisi comparativa tra l'approccio Numerico (Fasi 2-4) e l'approccio PINN (Fase 5).
---

# Fase 6: Confronto Finale (Numerico vs Deep Learning)

## Obiettivo
Il professore ha richiesto un confronto diretto tra le due metodologie. In questa fase conclusiva, estrapoleremo i risultati dai due "motori" del progetto per metterli uno contro l'altro.

## Metriche di Confronto
Scriveremo uno script `confronto_finale/compare_models.py` (e genereremo dei plot associati) per valutare i due approcci secondo i seguenti criteri:

1. **Accuratezza (Data Fitting):** 
   - L'algoritmo di Levenberg-Marquardt (Minimi Quadrati Classici) che valore di $D$ ha trovato?
   - La backpropagation della PINN che valore di $D$ ha trovato?
2. **Tempi Computazionali (Scalabilità):**
   - Quanto ci mette il solutore numerico a trovare $D$? (Meno iterazioni, ma ogni iterazione risolve un enorme sistema sparso 2500x2500).
   - Quanto ci mette la PINN ad addestrarsi? (Molte più iterazioni/epoche, calcolo dei gradienti su GPU).
3. **Robustezza ai Dati Sparsi:**
   - I metodi numerici su griglia richiedono spesso l'intero dominio definito. Le PINN possono essere addestrate anche se mancano enormi fette di dati nel dominio spaziale ("mesh-free data fitting").

## Deliverables
- Plot comparativi (`.png`) da inserire nella relazione accademica finale.
- Output a console formattati e leggibili.
