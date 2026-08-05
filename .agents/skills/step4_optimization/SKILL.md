---
name: Step 4 - Ottimizzazione e Minimi Quadrati
description: Istruzioni su come calibrare il parametro di diffusione usando i Minimi Quadrati Non Lineari.
---

# Fase 4: Data Fitting e Calibrazione (Optimization)
In questa fase il modello numerico incontra la realtà. Abbiamo il modello che predice $u_{simulata}(D)$ in funzione del coefficiente $D$, e i dati reali $u_{reale}$ della Protezione Civile (Fase 1).

## Funzione di Costo (Objective Function)
Dobbiamo minimizzare la differenza al quadrato tra previsione e realtà:
$J(D) = \sum_{i,j} (u_{simulata}^{i,j}(D) - u_{reale}^{i,j})^2$

Questa è la classica formulazione dei **Minimi Quadrati** (Least Squares).

## Ottimizzazione
Poiché la dipendenza di $u_{simulata}$ da $D$ non è lineare (richiede la soluzione del sistema lineare ad ogni step), questo è un problema di **Minimi Quadrati Non Lineari**.

## Librerie e Metodi
- Usare `scipy.optimize.least_squares`.
- Il metodo di ottimizzazione (es. `lm` per Levenberg-Marquardt o `trf` per Trust Region Reflective) testerà diversi valori di $D$, chiamando ripetutamente il nostro solutore (Fase 3), finché non trova il $D^*$ che fa assomigliare la simulazione ai dati veri il più possibile.
