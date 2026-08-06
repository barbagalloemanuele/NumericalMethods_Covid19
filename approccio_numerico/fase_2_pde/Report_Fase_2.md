# Report Fase 2: Discretizzazione della PDE e Algebra Lineare (Percorso A)

> **Nota di Contesto:** Questa fase fa parte del **Percorso A (Approccio Numerico Classico)** del progetto. Serve a costruire l'infrastruttura matematica su griglia che verrà poi confrontata con le Reti Neurali (Percorso B).


## Cosa bisogna fare in questa fase
Questa è la fase più "matematica" del progetto. Il modello epidemico si basa su un'Equazione alle Derivate Parziali (PDE) spaziale (l'equazione di diffusione). I computer non sanno risolvere equazioni continue; dobbiamo trasformarle in un gigantesco sistema lineare $Ax = b$.

Per farlo, usiamo il metodo delle **Differenze Finite**. Il Laplaciano spaziale 2D ($\nabla^2$) viene approssimato calcolando la differenza tra i contagi in un "pixel" e i suoi 4 pixel adiacenti (Nord, Sud, Est, Ovest). Questo procedimento genera una matrice quadrata enorme: la **Matrice A**.

## Cosa fa il codice (`pde_discretization.py`)
Lo script esegue l'operazione fondamentale: **costruisce la matrice A**.
Poiché abbiamo diviso l'Italia in una griglia $50 \times 50$, abbiamo $2500$ pixel totali. 
Questo significa che la nostra incognita $x$ sarà un vettore di $2500$ elementi, e la matrice $A$ sarà di dimensione $2500 \times 2500$ (cioè 6.250.000 elementi).
Se memorizzassimo questa matrice in modo classico (matrice densa), occuperemmo molta memoria inutilmente, perché quasi tutti i pixel NON sono collegati tra loro (il pixel di Milano influenza solo quello di Bergamo, non quello di Palermo).
Pertanto, il codice utilizza `scipy.sparse.diags` per creare una **matrice sparsa**, in cui vengono salvati in memoria *solo* gli elementi diversi da zero (le connessioni tra pixel adiacenti).

## Commento all'Output
Eseguendo lo script, l'output dimostra l'efficacia dei metodi numerici applicati:
```
Costruzione della matrice Laplaciana 2D per una griglia 50x50...
Dimensioni del sistema lineare (matrice A): (2500, 2500)
```
Il nostro sistema lineare $Ax=b$ ha 2500 equazioni e 2500 incognite.

```
Numero di elementi non nulli (nnz): 12300
Sparsità della matrice: 99.8032% (elementi nulli vs totali)
```
Questo è il dato cruciale per il corso del Prof. Boscarino. Su 6 milioni e rotti di celle nella matrice, **solo 12.300** sono diverse da zero. La matrice è vuota al 99.8%! Questo giustifica (e obbliga) l'uso dei **Metodi Iterativi** che implementeremo nella Fase 3, poiché i metodi diretti di base opererebbero su milioni di zeri sprecando tempo infinito.

```
Dimensioni del vettore incognito x e del termine noto b: (2500,)
```
Il vettore incognito (i contagi del giorno successivo per ogni pixel) e il termine noto hanno la corretta lunghezza di 2500. Il sistema è pronto per essere risolto.
