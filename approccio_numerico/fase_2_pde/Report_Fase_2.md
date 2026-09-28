# Report Fase 2: Discretizzazione della PDE e Algebra Lineare (Percorso A)

> **Nota di Contesto:** Questa fase fa parte del **Percorso A (Approccio Numerico Classico)** del progetto. Serve a costruire l'infrastruttura matematica su griglia che verrà poi confrontata con le Reti Neurali (Percorso B).


## Architettura della Cartella e Scelte Progettuali
Questa è la prima fase del *Percorso A* ed è isolata per trattare esclusivamente la costruzione del modello matematico, senza mescolarsi con la sua risoluzione:
- **`pde_discretization.py`**: Definisce il motore geometrico dell'equazione differenziale. La scelta di implementarlo come modulo separato (invece di inserirlo direttamente nei solutori) garantisce che il Laplaciano possa essere importato in modo modulare dalle fasi successive.
- **`Report_Fase_2.md`**: Questo documento esplicativo.

## Cosa bisogna fare in questa fase
Questa è la fase più "matematica" del progetto. Il modello epidemico si basa su un'Equazione alle Derivate Parziali (PDE) spaziale (l'equazione di diffusione). I computer non sanno risolvere equazioni continue; dobbiamo trasformarle in un gigantesco sistema lineare $Ax = b$.

## Fondamenti Matematici: Da Continuo a Discreto
Poiché il corso è "Numerical Methods for Scientific Computing", è fondamentale dominare la teoria matematica dietro al codice. Il nostro modello epidemico si basa su un'Equazione alle Derivate Parziali (PDE) di Diffusione:

$$ \frac{\partial u}{\partial t} = D \nabla^2 u $$

Dove $u$ è il numero di contagiati e $D$ è il coefficiente di diffusione. I computer non sanno risolvere equazioni continue; dobbiamo trasformarle in un gigantesco sistema lineare $Ax = b$. Per farlo, esaminiamo i tre concetti chiave:

### 1. Il Laplaciano Spaziale 2D ($\nabla^2$)
Il Laplaciano è un operatore matematico differenziale. In due dimensioni, è definito come la somma delle derivate parziali seconde spaziali:
$$ \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} $$
Fisicamente, la derivata seconda misura la "concavità". Il Laplaciano ci dice se in un punto c'è un accumulo (i vicini hanno più contagi, quindi il virus fluirà verso di noi) o una dispersione (noi abbiamo più contagi dei vicini, quindi il virus fluirà via da noi).

### 2. Le Differenze Finite (Sviluppo di Taylor)
Per tradurre la derivata seconda in algebra per il computer, la Analisi Numerica utilizza lo **Sviluppo in Serie di Taylor**. Se vogliamo calcolare la derivata seconda in un punto $x_i$, prendiamo il punto successivo ($x_{i+1}$) e il precedente ($x_{i-1}$). Troncando la serie di Taylor al secondo ordine, otteniamo l'approssimazione alle Differenze Finite Centrali:
$$ \frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1, j} - 2u_{i,j} + u_{i-1, j}}{\Delta x^2} $$
Questa formula calcola l'influenza dell'asse Est-Ovest. Facendo la stessa operazione sull'asse Y (Nord-Sud), otteniamo:
$$ \frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i, j+1} - 2u_{i,j} + u_{i, j-1}}{\Delta y^2} $$

### 3. Lo Stencil a 5 punti
Se assumiamo che la nostra griglia 50x50 abbia pixel perfettamente quadrati ($\Delta x = \Delta y = h$), possiamo sommare le due formule trovate sopra per ottenere l'approssimazione completa del Laplaciano:
$$ \nabla^2 u \approx \frac{u_{i+1, j} + u_{i-1, j} + u_{i, j+1} + u_{i, j-1} - 4u_{i,j}}{h^2} $$
Ecco la genesi della matematica del codice! Da questa formula, che unisce il punto centrale $(i,j)$ e i suoi 4 vicini diretti, nascono i famosi coefficienti della nostra matrice:
- **-4** per il nodo centrale $u_{i,j}$ (la diagonale principale `main_diag`)
- **+1** per i nodi adiacenti (le diagonali secondarie `off_diag_x` e `off_diag_y`).
Questa struttura geometrica a croce prende il nome tecnico di **Stencil a 5 punti**.

## Spiegazione del Codice Riga per Riga (`pde_discretization.py`)
Lo script traduce la mappa d'Italia (matematica continua) in una matrice calcolabile dal computer. Ecco cosa accade riga per riga:

- **`Riga 11: N = nx * ny`**
  Calcola il numero totale di pixel (50 * 50 = 2500). La nostra matrice finale $A$ sarà quindi quadrata e di dimensioni 2500x2500.
- **`Riga 14: main_diag = -4.0 * np.ones(N)`**
  Crea un array lungo 2500 pieno di `-4.0`. Questa sarà la *diagonale principale* della matrice. Nello Stencil a 5 punti, il -4 rappresenta il pixel "centrale" da cui i contagi defluiscono verso i vicini.
- **`Riga 17: off_diag_x = np.ones(N - 1)`**
  Crea un array lungo 2499 pieno di `1.0`. Queste sono le connessioni "Est e Ovest" (destra e sinistra) per ogni pixel.
- **`Riga 19: off_diag_x[nx-1::nx] = 0.0`**
  Questa riga risolve il "Teletrasporto". Poiché abbiamo appiattito una mappa 2D in un vettore 1D, il bordo Est della riga 1 finisce in memoria attaccato al bordo Ovest della riga 2. Questa riga prende l'ultimo pixel di ogni riga e annulla la connessione al pixel successivo imponendo uno `0.0`. In matematica accademica, questo si chiama imporre una **Condizione al contorno di Dirichlet nulla**.
- **`Riga 22: off_diag_y = np.ones(N - nx)`**
  Crea un array di `1.0` per le connessioni "Nord e Sud". Poiché la mappa è larga 50 pixel (`nx`), per guardare il vicino a Nord o a Sud il computer deve "saltare" di 50 posizioni nella memoria.
- **`Righe 25-26: diagonals = [...] e offsets = [...]`**
  Indica a Python la posizione esatta in cui montare le diagonali all'interno della futura matrice. L'offset `0` è il centro (i -4), gli offset `-1, 1` sono Est e Ovest, gli offset `-50, 50` sono Nord e Sud.
- **`Riga 29: A = sp.diags(diagonals, offsets, shape=(N, N), format='csr')`**
  Il momento in cui la matrice prende vita grazie alla compressione CSR, spiegata qui di seguito.

### Approfondimento: In cosa consiste il formato CSR?
CSR sta per **Compressed Sparse Row**. 
Una matrice normale di 2500x2500 ha 6.250.000 celle. La nostra equazione (lo stencil a 5 punti) piazza dei numeri solo in 12.300 celle, mentre restanti sei milioni di celle sono zeri inutili. 

Il formato CSR evita di salvare la griglia vuota. Invece, crea internamente tre piccole liste di numeri (vettori 1D):
1. **`data`**: Salva solo i numeri diversi da zero (i -4 e gli 1).
2. **`indices`**: Salva la colonna esatta in cui si trova ciascuno di questi numeri.
3. **`indptr`** (index pointer): Salva in quale punto del vettore `data` inizia una nuova riga della matrice.

In sintesi, il formato CSR dice al computer: *"Non disegnare un foglio enorme vuoto, ma annotati su un post-it solo quali numeri ci sono e in che colonna si trovano"*. Questo abbatte drasticamente il consumo di RAM e velocizza in modo mostruoso i calcoli della Fase 3.

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
