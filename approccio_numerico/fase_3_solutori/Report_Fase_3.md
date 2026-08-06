# Report Fase 3: Solutori Iterativi (Percorso A)

> **Nota di Contesto:** Questa fase fa parte del **Percorso A (Approccio Numerico Classico)** del progetto. Applica i metodi iterativi per risolvere il sistema generato nella Fase 2.


## Cosa bisogna fare in questa fase
Avendo costruito l'enorme matrice sparsa $A$ (Laplaciano) nella Fase 2, dobbiamo risolvere il sistema lineare $Ax = b$ per poter simulare lo scorrere del tempo (e la propagazione del virus). 
Trattandosi di una matrice per il 99.8% vuota, l'uso di metodi "diretti" (come la fattorizzazione LU o l'inversione di matrice) è assolutamente da evitare, pena l'esaurimento della RAM e tempi di calcolo biblici.

Implementiamo quindi i **Metodi Iterativi** studiati nel corso del Prof. Boscarino. Essi non "invertono" la matrice, ma partono da un'ipotesi iniziale e la raffinano iterativamente finché l'errore (il residuo $||Ax - b||$) non scende sotto una certa tolleranza ($\text{tol} = 10^{-6}$).

## Cosa fa il codice (`solvers.py`)
Lo script esegue e confronta due dei più famosi solutori iterativi disponibili in `scipy.sparse.linalg`:
1. **CG (Conjugate Gradient - Gradiente Coniugato):** È il metodo "re" per eccellenza quando la matrice è Simmetrica e Definita Positiva (SPD). Dato che il nostro Laplaciano ha i -4 sulla diagonale principale, lo moltiplichiamo per -1 per renderlo definito positivo e applicare il CG.
2. **GMRES (Generalized Minimal Residual):** È un metodo più flessibile, capace di risolvere anche sistemi non simmetrici. È utile averlo implementato come benchmark o nel caso in cui, in futuro, si aggiunga al modello un termine di convezione spaziale (es. il vento che trasporta inquinamento) che romperebbe la simmetria della matrice.

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
Tempo di calcolo: 0.0324 secondi.
Residuo relativo finale: 9.78e-07
```
Il GMRES, essendo un metodo più generalizzato e "pesante", ci impiega più iterazioni ($418$) e più tempo ($32.4$ millisecondi), dimostrando empiricamente la teoria vista a lezione: **se la matrice è Simmetrica e Definita Positiva, il CG vince sempre a mani basse.**

Le soluzioni calcolate dai due metodi sono numericamente equivalenti (la differenza è dell'ordine di $10^{-3}$), confermando la bontà dell'implementazione.

## Grafici di Convergenza (`visualize.py`)
In questa cartella è presente anche lo script `visualize.py`. Il suo scopo è generare un grafico in scala semilogaritmica (standard in Analisi Numerica) che mostra visivamente quanto discusso sopra: il crollo del residuo iterazione dopo iterazione per il CG e per il GMRES. L'immagine in output (`convergenza_solutori.png`) è pronta per essere allegata alla Relazione finale.
