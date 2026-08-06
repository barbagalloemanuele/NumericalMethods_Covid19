---
name: Fase 5 - Physics-Informed Neural Networks (PINN)
description: Istruzioni per l'implementazione del Percorso B. Addestramento di una PINN in PyTorch per modellare l'equazione di diffusione spaziale e fare data fitting.
---

# Fase 5: Physics-Informed Neural Networks (PINN)

## Obiettivo
A differenza dell'approccio numerico puro (Fasi 2-4) che discretizza l'equazione con differenze finite su una griglia, la PINN usa una **Rete Neurale Artificiale** come approssimatore universale di funzione continuo.
La rete neurale prenderà in input le coordinate spaziali $(x, y)$ e temporali $(t)$, e restituirà i contagi $u(x, y, t)$.

## Architettura del Modello
- **Libreria:** PyTorch.
- **Struttura:** Multi-Layer Perceptron (MLP) fully connected.
- **Input:** 3 neuroni $(x, y, t)$.
- **Output:** 1 neurone $u$.
- **Parametri Addestrabili Extra:** Il coefficiente di diffusione $D$ (che inizializziamo a un valore casuale, es. 1.0) diventerà un `nn.Parameter` ottimizzabile durante la backpropagation.

## La Funzione di Loss Multi-Obiettivo
Il cuore della PINN è la funzione di costo, composta da due termini sommati:
1. **Data Loss (MSE_data):** Misura l'errore tra le predizioni della rete neurale e i dati reali osservati (simulando l'estrazione della Protezione Civile).
2. **Physics Loss (MSE_physics):** Penalizza le soluzioni che non rispettano l'equazione differenziale. Calcoliamo le derivate della rete rispetto agli input usando `torch.autograd.grad`. 
   L'equazione residua è: $f = u_t - D(u_{xx} + u_{yy})$. 
   Vogliamo che $f \approx 0$ su tutti i punti del dominio spaziale.

## Workflow di Sviluppo
1. Sviluppare il codice in `approccio_pinn/fase_5_pinn/pinn_solver.py`.
2. Testare in locale su piccoli batch usando la CPU del Mac (o MPS).
3. Una volta debuggato, sincronizzare con `rsync` sul cluster UNICT e lanciare il training su GPU via SLURM/Apptainer per ottenere il valore di $D$ addestrato.
