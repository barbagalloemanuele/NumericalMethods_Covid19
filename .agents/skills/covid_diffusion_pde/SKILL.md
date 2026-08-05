---
name: Modello Spaziale COVID-19 (Metodi Numerici)
description: Contesto e linee guida per lo sviluppo del progetto di Metodi Numerici sul Calcolo Scientifico (diffusione spaziale epidemica con PDE, Metodi Iterativi e Minimi Quadrati).
---

# Contesto del Progetto
L'utente Emanuele sta sviluppando il progetto finale per l'esame di "Numerical Methods for Scientific Computing" (Prof. Boscarino). 
Il progetto modella la diffusione spaziale del COVID-19 in Italia utilizzando un'Equazione alle Derivate Parziali (PDE).

# Obiettivi e Requisiti Teorici
Il progetto deve dimostrare padronanza nei seguenti argomenti del corso:
1. **Discretizzazione di PDE**: Uso delle differenze finite per discretizzare lo spazio geografico (es. mappa dell'Italia divisa in griglia).
2. **Sistemi Lineari e Metodi Iterativi**: La discretizzazione porta a un sistema lineare grande e sparso ($Ax=b$). Devono essere implementati, testati e confrontati metodi iterativi come **Conjugate Gradient (CG)** e **GMRES** per risolverlo.
3. **Ottimizzazione e Minimi Quadrati**: Uso dei dati storici spaziali della Protezione Civile (casi per provincia) per effettuare il *data fitting*. Bisogna usare i Minimi Quadrati per calibrare i parametri del modello (come il coefficiente di diffusione) in modo da minimizzare l'errore tra i dati simulati e quelli reali.

# Strumenti e Dati
- **Linguaggio:** Python
- **Librerie:** NumPy (per le matrici e algebra lineare di base), SciPy (per strutture dati sparse e metodi iterativi di riferimento), Matplotlib (per visualizzazione mappe e grafici di convergenza).
- **Dati:** Repository GitHub della Protezione Civile Italiana (dati provinciali COVID-19).

# Regole di Comportamento
- Quando scrivi codice per risolvere $Ax=b$, **non usare metodi diretti** (come `np.linalg.solve`), ma implementa o utilizza metodi iterativi (CG/GMRES) enfatizzando lo studio di convergenza e tolleranza, poiché è ciò su cui verte il corso.
- Concentrati sulla natura "sparsa" delle matrici. Usa `scipy.sparse`.
- Qualsiasi script deve contenere un'attenta misurazione dei tempi e analisi del residuo per produrre i benchmark richiesti dalla relazione finale.
