# Report Fase 3: Solutori Iterativi (Percorso A)

> **Nota di Contesto:** Questa fase fa parte del **Percorso A (Approccio Numerico Classico)** del progetto. Applica i metodi iterativi per risolvere il sistema generato nella Fase 2.


## Architettura della Cartella e Scelte Progettuali
Questa cartella racchiude il motore risolutivo dell'equazione differenziale.
- **`solvers.py`**: È lo script di "core computation" che importa la matrice dalla Fase 2 e implementa gli algoritmi iterativi. La scelta è caduta su CG e GMRES (escludendo metodi esatti) proprio per via dell'enorme sparsità della matrice.
- **`visualize.py`**: Separato intenzionalmente da `solvers.py`. Questa scelta garantisce che i pesanti calcoli lineari non siano accoppiati al rendering grafico, permettendo di importare le funzioni del solver in ambienti senza interfaccia grafica (es. un cluster) senza crash legati a Matplotlib.
- **`convergenza_solutori.png`**: L'output grafico generato da `visualize.py`, mantenuto qui per dimostrare visivamente la bontà dei due solutori, essenziale per la documentazione finale.
- **`Report_Fase_3.md`**: Questo documento.

## Fondamenti Matematici: I Solutori Iterativi e i Sottospazi di Krylov
Poiché la matrice $A$ del Laplaciano è gigantesca ($2500 \times 2500$) ma estremamente vuota (99.8% di zeri, formato CSR), l'approccio classico per risolvere il sistema $Ax = b$ tramite l'inversa ($x = A^{-1}b$) fallirebbe miseramente.
In Analisi Numerica, calcolare l'inversa di una matrice sparsa provoca il fenomeno del **Fill-in**: i buchi vuoti si riempiono di numeri, distruggendo il formato CSR e saturando la RAM.

La soluzione risiede nei **Metodi Iterativi basati sui Sottospazi di Krylov**. Essi non calcolano mai l'inversa di $A$. Partendo da un'ipotesi iniziale $x_0$, calcolano il residuo iniziale $r_0 = b - Ax_0$. 
Matematicamente, uno Spazio di Krylov di ordine $m$ è lo spazio vettoriale generato dalle moltiplicazioni successive della matrice $A$ per il residuo $r_0$:
$$ \mathcal{K}_m(A, r_0) = \text{span} \{ r_0, Ar_0, A^2r_0, \dots, A^{m-1}r_0 \} $$
Il grande vantaggio computazionale è che questi metodi richiedono **solo moltiplicazioni Matrice-Vettore**, operazione in cui il formato CSR è imbattibile.

### Perché abbiamo scelto CG e GMRES?
Nel progetto abbiamo implementato e confrontato due specifici solutori di Krylov per precisi motivi accademici:

1. **Gradiente Coniugato (CG)**
   - **Come funziona:** Il CG cerca la soluzione ottimale all'interno dello Spazio di Krylov costruendo direzioni di ricerca $p_k$ che sono *A-ortogonali* (o coniugate), ovvero $p_i^T A p_j = 0$. Questo garantisce convergenza rapida.
   - **Requisiti:** Funziona *solo* se la matrice $A$ è Simmetrica e Definita Positiva (SPD).
   - **Perché l'abbiamo scelto:** Il Laplaciano standard 2D (moltiplicato per -1) è SPD. Il CG è dimostrato essere l'algoritmo di Krylov ottimale (minimo tempo e memoria) per le matrici SPD. È la nostra "prima scelta" per le massime prestazioni.

2. **GMRES (Generalized Minimal Residual)**
   - **Come funziona:** A differenza del CG, il GMRES usa l'iterazione di Arnoldi per costruire una base ortonormale dello spazio di Krylov. Ad ogni passo $m$, cerca il vettore $x_m \in \mathcal{K}_m$ che minimizza rigorosamente la norma del residuo $||b - Ax_m||_2$.
   - **Requisiti:** Nessuno. Funziona per qualsiasi matrice, anche asimmetrica.
   - **Perché l'abbiamo scelto:** È il nostro benchmark di robustezza. Se volessimo espandere il progetto aggiungendo un termine di convezione spaziale (es. il "vento" o lo spostamento pendolare da una provincia all'altra), la matrice $A$ diventerebbe **asimmetrica**. Il CG andrebbe in crash, ma il GMRES continuerebbe a convergere perfettamente. Ingegneristicamente, garantisce la scalabilità futura del modello.

## Spiegazione del Codice Riga per Riga (`solvers.py`)
Lo script esegue il confronto matematico tra questi due giganti dell'Algebra Lineare Computazionale:

- **`Righe 18-22: def callback_cg / gmres`**
  Queste funzioni agiscono da "spie". Vengono invocate da Python alla fine di ogni singola iterazione dell'algoritmo. Ci permettono di salvare l'errore (residuo) passo dopo passo per poter disegnare il grafico di convergenza.
- **`Riga 28: x, info = spla.cg(A, b, rtol=tol, atol=0, callback=callback_cg)`**
  Avvia il risolutore del Gradiente Coniugato. `rtol` imposta la tolleranza relativa ($10^{-6}$). La variabile `x` conterrà la soluzione finale, mentre `info` ci dice se c'è stato un errore.
- **`Riga 31: x, info = spla.gmres(A, b, rtol=tol, atol=0, callback=callback_gmres...)`**
  Avvia il risolutore GMRES per il confronto.
- **`Riga 45: final_residual = np.linalg.norm(b - A.dot(x)) / np.linalg.norm(b)`**
  Questa è l'implementazione in codice della formula matematica del Residuo Relativo: $\frac{||b - Ax||}{||b||}$. Serve a verificare empiricamente che il solutore non ci stia mentendo.
- **`Riga 55: A = -build_laplacian_2d(nx, ny)`**
  **Dettaglio teorico cruciale:** Il Laplaciano generato nella Fase 2 ha dei `-4` sulla diagonale. In matematica, questo lo rende *Definito Negativo*. Se lo passassimo al CG, l'algoritmo andrebbe in crash perché richiede una matrice *Definita Positiva*. Moltiplicando la matrice intera per `-1`, invertiamo i segni (i -4 diventano +4) e la trasformiamo magicamente in una **Matrice SPD**, salvando il calcolo!
- **`Riga 60: b = np.random.rand(N)`**
  Per testare la pura potenza dei solutori, non passiamo subito i dati reali del Covid, ma generiamo un vettore di rumore casuale di 2500 elementi. Se i solutori riescono a risolvere il caos casuale, risolveranno sicuramente i dati reali.

## Q&A per la Presentazione (Approfondimenti Teorici)

**1. Perché il formato CSR è imbattibile nelle moltiplicazioni matrice-vettore?**
In una moltiplicazione standard riga-per-colonna, la CPU deve moltiplicare ogni cella della matrice per il vettore. Se la nostra matrice 2500x2500 è vuota al 99.8%, la CPU perde tempo a calcolare 6 milioni di volte l'operazione $0 \times \text{vettore} = 0$. Il formato CSR ha in memoria solo le coordinate delle 12.300 celle piene. Durante la moltiplicazione, salta letteralmente tutti gli zeri, riducendo lo sforzo computazionale di quasi mille volte.

**2. Cosa significa Matrice Simmetrica e Definita Positiva (SPD)? (Esempio)**
- **Simmetrica ($A = A^T$):** Significa che lo scambio tra due punti è bidirezionale e identico. Esempio: Il tasso con cui i cittadini di Prato si recano a Firenze è esattamente identico al tasso con cui i cittadini di Firenze vanno a Prato. La connessione Prato-Firenze ($A_{i,j}$) è uguale alla connessione Firenze-Prato ($A_{j,i}$).
- **Definita Positiva ($x^T A x > 0$):** Fisicamente significa che il sistema "dissipa" o stabilizza l'energia, non la crea dal nulla. In un'epidemia senza nuovi focolai esterni, il virus tende a diffondersi fino ad appiattirsi; non può generare un'esplosione spontanea infinita di casi dal nulla.

**3. Perché se si aggiunge "il vento" la matrice diventa asimmetrica? (Esempio)**
Il "vento" in fluidodinamica si chiama **Convezione**. Immagina che ci sia un fortissimo vento (o flusso pendolare a senso unico) che va da Ovest verso Est. Il virus viaggerà benissimo da Prato (Ovest) verso Firenze (Est), ma farà un'enorme fatica a tornare indietro controvento. Matematicamente, il contagio Prato $\rightarrow$ Firenze varrà $2.0$, ma il contagio Firenze $\rightarrow$ Prato varrà solo $0.5$. La cella $A_{i,j}$ non è più uguale alla cella $A_{j,i}$. La matrice perde la simmetria, smette di essere SPD, e il Gradiente Coniugato (CG) va in crash. Ecco perché abbiamo introdotto il GMRES: funzionerebbe anche col vento.

**4. Riga 28: Cosa rappresenta `rtol = 1e-6` e chi l'ha deciso?**
Sta per *Relative Tolerance* (Tolleranza Relativa). I metodi iterativi potrebbero continuare all'infinito, quindi dobbiamo dirgli quando fermarsi. Abbiamo scelto $10^{-6}$ (un milionesimo) per tre precisi motivi di Analisi Numerica:
1. **Inutilità dell'eccesso (Over-solving):** I nostri dati di partenza (i tamponi Covid) sono altamente imprecisi e "rumorosi". Calcolare la matematica al quindicesimo decimale ($10^{-15}$) non renderà la previsione medica più esatta, sprecherà solo ore di CPU per rincorrere l'illusione della perfezione algebrica su dati fisici imperfetti.
2. **Precisione di Macchina:** I computer usano numeri "float64" che hanno un limite fisico di precisione (circa 15 decimali). Avvicinandosi a quel limite, subentrano errori di arrotondamento che paradossalmente *peggiorano* la soluzione.
3. **Lo Standard Accademico:** $10^{-6}$ è universalmente riconosciuto nei calcoli delle PDE come il perfetto "sweet spot": garantisce che l'errore introdotto dal computer sia invisibile rispetto all'entità del problema, pur bloccando l'algoritmo in frazioni di secondo.

**5. Riga 45: Cos'è il Residuo Relativo e cosa significa?**
La formula è $\frac{||b - Ax||}{||b||}$. 
- $b$ è il nostro bersaglio.
- $Ax$ è il punto in cui siamo arrivati.
- $b - Ax$ è l'errore assoluto (la distanza dal bersaglio).
Perché dividerlo per $b$ (rendendolo *relativo*)? Se l'errore assoluto è di $100$ malati, è grave? Dipende. Se in tutta Italia ci sono $1.000.000$ di malati, sbagliare di $100$ è una sciocchezza (lo 0.01%). Ma se in Italia ci sono $100$ malati, sbagliare di $100$ significa aver sbagliato del 100%! Il residuo relativo valuta la gravità dell'errore in proporzione alla grandezza del bersaglio reale.

**6. Stiamo testando su numeri casuali o sui numeri del Covid? Quali sono quelli reali?**
In questa Fase 3 stiamo collaudando il "Motore dell'auto" (i solutori) sul banco di prova, usando numeri casuali (`np.random.rand`). Perché? Per dimostrare scientificamente che l'algoritmo di calcolo matriciale funziona alla perfezione su qualsiasi sistema (matematica pura). 
I numeri del Covid veri (che sono le 10.379 righe estratte nella Fase 1) verranno inseriti nella **Fase 4**. Lì prenderemo questo motore collaudato e lo monteremo nell'algoritmo di Ottimizzazione per inseguire la vera curva epidemica.

**7. Cos'è l'Iterazione di Arnoldi e cosa significa costruire una base ortonormale?**
Per capire Arnoldi, dobbiamo capire il problema che risolve. Nello Spazio di Krylov continuiamo a moltiplicare la matrice per lo stesso vettore ($r_0, Ar_0, A^2r_0...$). Il problema algebrico è che, facendo queste moltiplicazioni ripetute, i nuovi vettori tendono ad allinearsi tutti nella stessa direzione (diventano quasi paralleli). In matematica, usare vettori paralleli per fare calcoli causa errori di arrotondamento catastrofici per il computer.
- **Base Ortonormale:** Significa creare un set di assi di riferimento (come X, Y e Z) che siano perfettamente perpendicolari tra loro (*ortogonali*) e lunghi esattamente 1 (*normali*). È il sistema di coordinate più stabile e perfetto che esista per fare calcoli senza far impazzire la CPU.
- **L'Iterazione di Arnoldi:** È l'algoritmo "muratore" del GMRES (basato sul processo di Gram-Schmidt). Ogni volta che il GMRES genera un nuovo vettore di Krylov, l'iterazione di Arnoldi interviene: prende questo nuovo vettore, lo raddrizza costringendolo a essere perfettamente perpendicolare a tutti quelli calcolati in precedenza, e lo taglia a lunghezza 1. In questo modo, l'algoritmo Arnoldi prende il caos dei vettori di Krylov e costruisce un'impalcatura (la base ortonormale) perfetta e stabile, permettendo al GMRES di trovare la soluzione sicura anche se la matrice è asimmetrica.

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

## Interpretazione del Grafico (`convergenza_solutori.png`)
In questa cartella è presente lo script `visualize.py` che genera il grafico `convergenza_solutori.png`. Se all'esame il professore ti chiede di spiegare questa immagine, ecco cosa devi dirgli:

1. **La Scala Semilogaritmica (Asse Y):** 
   L'asse verticale rappresenta il Residuo Relativo (l'errore), ma non è in scala normale. È in **scala logaritmica** ($10^0, 10^{-1}, \dots, 10^{-6}$). Usiamo questa scala perché l'errore precipita in modo esponenziale. Se usassimo una scala normale, vedremmo solo una linea a forma di L schiacciata sul fondo. La scala logaritmica ci permette di vedere la "pendenza" reale della discesa matematica verso la soluzione perfetta. L'asse X è invece lineare (numero di iterazioni).
2. **Il crollo del CG (La Ferrari):** 
   La curva del Gradiente Coniugato scende in modo estremamente ripido. Questa è la **dimostrazione visiva del teorema matematico**: per matrici SPD, il CG abbatte l'errore molto più velocemente di qualsiasi altro metodo iterativo. In poco più di 100 iterazioni, sfonda il "pavimento" della tolleranza (la linea tratteggiata orizzontale a $10^{-6}$) e il calcolo si ferma.
3. **La discesa più dolce del GMRES (Il Fuoristrada):** 
   La curva del GMRES scende anch'essa, dimostrando che la matrice converge, ma lo fa con una pendenza molto più dolce. Ha bisogno di oltre 400 iterazioni (spostandosi molto più a destra sull'asse X) per raggiungere lo stesso livello di precisione. Inoltre, internamente, ogni iterazione del GMRES costa di più in termini di CPU rispetto al CG perché deve costruire la famosa *Base Ortonormale di Arnoldi*.

In conclusione, questo grafico è la prova provata che il nostro modello matematico (matrice SPD) è perfettamente abbinato al solutore più efficiente in circolazione (il CG). L'immagine è fondamentale e andrà incollata direttamente nelle slide della tua presentazione finale.
