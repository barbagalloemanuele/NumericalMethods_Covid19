# Report Fase 3: Solutori Iterativi (Percorso A)

> **Nota di Contesto:** Questa fase fa parte del **Percorso A (Approccio Numerico Classico)** del progetto. Applica i metodi iterativi per risolvere il sistema generato nella Fase 2.


## Architettura della Cartella e Scelte Progettuali
Questa cartella racchiude il motore risolutivo dell'equazione differenziale.
- **`solvers.py`**: È lo script di "core computation" che importa la matrice dalla Fase 2 e implementa gli algoritmi iterativi. La scelta è caduta su CG e GMRES (escludendo metodi esatti) proprio per via dell'enorme sparsità della matrice.
- **`visualize.py`**: Separato intenzionalmente da `solvers.py`. Questa scelta garantisce che i pesanti calcoli lineari non siano accoppiati al rendering grafico, permettendo di importare le funzioni del solver in ambienti senza interfaccia grafica (es. un cluster) senza crash legati a Matplotlib.
- **`convergenza_solutori.png`**: L'output grafico generato da `visualize.py`, mantenuto qui per dimostrare visivamente la bontà dei due solutori, essenziale per la documentazione finale.
- **`Report_Fase_3.md`**: Questo documento.

## Cosa bisogna fare in questa fase
Avendo costruito l'enorme matrice sparsa $A$ (Laplaciano) nella Fase 2, dobbiamo risolvere il sistema lineare $Ax = b$ per poter simulare lo scorrere del tempo (e la propagazione del virus). 
Trattandosi di una matrice per il 99.8% vuota, l'uso di metodi "diretti" (come la fattorizzazione LU o l'inversione di matrice) è assolutamente da evitare, pena l'esaurimento della RAM e tempi di calcolo biblici.

Implementiamo quindi i **Metodi Iterativi** studiati nel corso del Prof. Boscarino. Essi non "invertono" la matrice, ma partono da un'ipotesi iniziale e la raffinano iterativamente finché l'errore (il residuo $||Ax - b||$) non scende sotto una certa tolleranza ($\text{tol} = 10^{-6}$).

## Spiegazione Dettagliata del Codice (`solvers.py`)
Lo script mette alla prova due giganti della matematica computazionale importati da `scipy.sparse.linalg`:

### 1. Preparazione del Sistema (SPD)
- **`A = -build_laplacian_2d(nx, ny)`**: Il Laplaciano standard ha valori negativi sulla diagonale principale. Per usare i metodi di discesa più potenti, la matrice deve essere Simmetrica e Definita Positiva (SPD). Moltiplicandola per -1 risolviamo il problema algebrico senza intaccare il significato fisico (basta ricordarsene nella Fase 4).
- **`b = np.random.rand(N)`**: Generiamo un termine noto temporaneo di dimensione 2500 per testare matematicamente l'efficienza dei solutori prima di accoppiarli ai dati reali.

### 2. Risoluzione e Callbacks (`solve_system`)
- **Il Meccanismo dei Callback**: La funzione `solve_system` non si limita a dire "risolvi", ma "spia" l'algoritmo. Passando `callback=callback_cg` e `callback=callback_gmres` ai solutori di SciPy, il programma registra un dato alla fine di *ogni singola iterazione*, permettendoci di tracciare la curva di convergenza.
- **Gradiente Coniugato (`spla.cg`)**: Ottimale per matrici SPD. Cerca la soluzione viaggiando lungo direzioni "coniugate" (ortogonali rispetto alla matrice A).
- **GMRES (`spla.gmres`)**: Il *Generalized Minimal Residual* è un'alternativa formidabile. Non richiede che la matrice sia SPD. L'abbiamo implementato perché se, in futuro, si volesse aggiungere alla simulazione il vento (termine convettivo spaziale), la matrice perderebbe la simmetria e il CG smetterebbe di funzionare.

## Commento all'Output
Eseguendo lo script, l'output mostra la potenza dei metodi iterativi:
```
Risoluzione del sistema di dimensione 2500x2500...

--- Risoluzione tramite CG ---
Convergenza raggiunta in 127 iterazioni.
Tempo di calcolo: 0.0028 secondi.
Residuo relativo finale: 9.11e-07
```
Il Gradiente Coniugato si dimostra **fulmineo**. Risolve un sistema di $2500 \times 2500$ in soli $2.8$ millisecondi, impiegando appena $127$ iterazioni per far scendere l'errore sotto la soglia di $10^{-6}$.

```
--- Risoluzione tramite GMRES ---
Convergenza raggiunta in 418 iterazioni.
Tempo di calcolo: 0.0326 secondi.
Residuo relativo finale: 9.78e-07

Differenza assoluta tra le soluzioni dei due metodi: 3.23e-03
```
Il GMRES, essendo un metodo più generalizzato e "pesante", ci impiega più iterazioni ($418$) e più tempo ($32.6$ millisecondi), dimostrando empiricamente la teoria vista a lezione: **se la matrice è Simmetrica e Definita Positiva, il CG vince sempre a mani basse.**

Le soluzioni calcolate dai due metodi sono numericamente equivalenti (la differenza assoluta, misurata, è dell'ordine di $10^{-3}$), confermando la bontà dell'implementazione.

## Grafici di Convergenza (`visualize.py`)
In questa cartella è presente anche lo script `visualize.py`. Il suo scopo è generare un grafico in scala semilogaritmica (standard in Analisi Numerica) che mostra visivamente quanto discusso sopra: il crollo del residuo iterazione dopo iterazione per il CG e per il GMRES. L'immagine in output (`convergenza_solutori.png`) è pronta per essere allegata alla Relazione finale.
