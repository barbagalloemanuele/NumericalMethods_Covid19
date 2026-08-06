---
name: Step 3 - Solutori Iterativi
description: Istruzioni sull'implementazione dei metodi iterativi per sistemi lineari sparsi (CG, GMRES).
---

# Fase 3: Solutori Iterativi (Numerical Methods)
Una volta ottenuto il grande sistema lineare $Ax = b$ dalla discretizzazione (Fase 2), bisogna risolverlo in modo efficiente.

## Vincoli di Implementazione
- **Vietato usare metodi diretti** come `np.linalg.solve` o l'inversione della matrice. 
- Il corso di Numerical Methods verte sull'utilizzo di Metodi Iterativi per matrici sparse.

## Metodi da Utilizzare
1. **Gradiente Coniugato (CG):** Ideale perché la nostra matrice del Laplaciano (negata) è Simmetrica e Definita Positiva (SPD).
2. **GMRES:** Un'alternativa robusta nel caso introducessimo termini convettivi che rompono la simmetria.

## Librerie
Usare `scipy.sparse.linalg.cg` e `scipy.sparse.linalg.gmres`.

## Task di Analisi
Il codice deve includere:
- Una misurazione del **tempo computazionale**.
- Un tracciamento del **residuo relativo** ($||Ax_k - b|| / ||b||$) ad ogni iterazione, per poi plottarlo.
- La definizione esplicita di una tolleranza (es. `tol = 1e-6`).
