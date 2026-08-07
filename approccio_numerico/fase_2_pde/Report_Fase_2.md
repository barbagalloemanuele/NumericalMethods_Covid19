# Report Fase 2: Discretizzazione della PDE e Algebra Lineare (Percorso A)

> **Nota di Contesto:** Questa fase fa parte del **Percorso A (Approccio Numerico Classico)** del progetto. Serve a costruire l'infrastruttura matematica su griglia che verrà poi confrontata con le Reti Neurali (Percorso B).


## Architettura della Cartella e Scelte Progettuali
Questa è la prima fase del *Percorso A* ed è isolata per trattare esclusivamente la costruzione del modello matematico, senza mescolarsi con la sua risoluzione:
- **`pde_discretization.py`**: Definisce il motore geometrico dell'equazione differenziale. La scelta di implementarlo come modulo separato (invece di inserirlo direttamente nei solutori) garantisce che il Laplaciano possa essere importato in modo modulare dalle fasi successive.
- **`Report_Fase_2.md`**: Questo documento esplicativo.

## Cosa bisogna fare in questa fase
Questa è la fase più "matematica" del progetto. Il modello epidemico si basa su un'Equazione alle Derivate Parziali (PDE) spaziale (l'equazione di diffusione). I computer non sanno risolvere equazioni continue; dobbiamo trasformarle in un gigantesco sistema lineare $Ax = b$.

Per farlo, usiamo il metodo delle **Differenze Finite**. Il Laplaciano spaziale 2D ($\nabla^2$) viene approssimato calcolando la differenza tra i contagi in un "pixel" e i suoi 4 pixel adiacenti (Nord, Sud, Est, Ovest). Questo procedimento genera una matrice quadrata enorme: la **Matrice A**.

## Spiegazione Dettagliata del Codice (`pde_discretization.py`)
Lo script esegue l'operazione fondamentale dell'approccio numerico: **costruisce la matrice A** (Laplaciano) evitando memory overflow tramite la libreria `scipy.sparse`.

### La costruzione delle Diagonali
Il Laplaciano 2D a differenze finite (stencil a 5 punti) si traduce in una matrice enorme in cui le uniche informazioni utili giacciono su 5 diagonali specifiche:
- **`main_diag = -4.0 * np.ones(N)`**: La diagonale principale contiene tutti -4. Rappresenta il decadimento o la perdita di concentrazione del virus dal punto centrale verso le celle adiacenti.
- **`off_diag_x = np.ones(N - 1)`**: Le diagonali immediatamente sopra e sotto quella principale contengono gli 1. Rappresentano lo scambio col vicino di Destra e di Sinistra sull'asse X.
- **`off_diag_x[nx-1::nx] = 0.0`**: Questo dettaglio di codice è fondamentale. Poiché abbiamo appiattito una griglia 2D in un vettore 1D (riordinamento lessicografico), il "bordo destro" della riga 1 non confina col "bordo sinistro" della riga 2. Inserire gli zero annulla matematicamente il contagio "attraverso il bordo della mappa".
- **`off_diag_y = np.ones(N - nx)`**: Queste diagonali, posizionate a distanza $nx$ dalla principale, connettono un pixel col suo vicino a Nord e a Sud (asse Y).

### Assemblaggio Sparso (`sp.diags`)
- **`A = sp.diags(diagonals, offsets, shape=(N, N), format='csr')`**: Invece di creare una matrice di zeri grande $2500 \times 2500$ e inserire i numeri (che consumerebbe gigabyte di RAM se la mappa fosse più fitta), ordiniamo al computer di allocare in memoria **esclusivamente** le 5 diagonali, "comprimendo" tutto il resto in formato CSR (*Compressed Sparse Row*). Il formato CSR è lo standard aureo per far volare le prestazioni dei solutori iterativi nella fase successiva.

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
