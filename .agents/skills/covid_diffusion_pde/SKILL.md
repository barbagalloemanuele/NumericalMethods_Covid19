---
name: Modello Spaziale COVID-19 (Numerico vs PINN)
description: Contesto e linee guida per lo sviluppo del progetto di Metodi Numerici sul Calcolo Scientifico (diffusione spaziale epidemica con PDE Classiche vs Physics-Informed Neural Networks).
---

# Contesto del Progetto
L'utente Emanuele sta sviluppando il progetto finale per l'esame di "Numerical Methods for Scientific Computing" (Prof. Boscarino). 
Il progetto modella la diffusione spaziale del COVID-19 in Italia e, su esplicita richiesta del professore, effettua un **confronto** tra un approccio prettamente numerico (matrici sparse) e uno basato sul Deep Learning (PINN).

# Obiettivi e Requisiti Teorici
Il progetto deve dimostrare padronanza in due grandi ecosistemi del calcolo scientifico:

## Percorso A: Approccio Numerico Classico
1. **Discretizzazione di PDE**: Uso delle differenze finite per discretizzare lo spazio geografico.
2. **Sistemi Lineari e Metodi Iterativi**: La discretizzazione porta a un sistema lineare grande e sparso ($Ax=b$). Si usano **CG** e **GMRES** per risolverlo, misurandone la convergenza.
3. **Ottimizzazione e Minimi Quadrati**: Uso dell'algoritmo di Levenberg-Marquardt per effettuare il *data fitting* sul parametro incognito $D$ (coefficiente di diffusione).

## Percorso B: Approccio PINN (Physics-Informed Neural Networks)
1. **Deep Learning per PDEs**: Invece di matrici, si usa una Rete Neurale Artificiale (MLP in PyTorch) per approssimare la soluzione $u(x,y,t)$.
2. **Loss Ingegnerizzata**: La rete apprende calcolando i gradienti (`autograd`) della funzione di Loss. La Loss è la somma dell'errore sui dati (Data Loss) e del residuo dell'equazione differenziale (Physics Loss).
3. **Scoperta Parametri**: Il coefficiente $D$ viene estratto addestrandolo come peso della rete stessa.

# Strumenti e Dati
- **Linguaggio:** Python
- **Librerie Percorso A:** NumPy, SciPy (per matrici sparse e solutori iterativi).
- **Librerie Percorso B:** PyTorch (per la PINN).
- **Ambiente di Esecuzione:** Mac (sviluppo/debug locale) e GPU Cluster UNICT (per l'addestramento intensivo tramite SLURM e Apptainer).
- **Dati:** Repository GitHub della Protezione Civile Italiana (dati provinciali COVID-19 - Prima Ondata Feb/Mag 2020).

# Regole di Comportamento
- Mantenere la netta separazione tra `approccio_numerico` e `approccio_pinn`.
- Per le Fasi Numeriche: **non usare metodi diretti** (come `np.linalg.solve`), usa iterativi. Concentrati sulla sparsità.
- Il fine ultimo è il **Confronto Finale (Fase 6)**: ogni script deve misurare i propri tempi, iterazioni ed errori, per poter produrre i grafici comparativi conclusivi.
