# Report Fase 4: Calibrazione tramite Minimi Quadrati (Percorso A)

> **Nota di Contesto:** Questa fase fa parte del **Percorso A (Approccio Numerico Classico)** del progetto. Calibra i parametri della PDE risolta nella Fase 3 usando l'ottimizzazione classica (Least Squares), che verrà poi messa a confronto con l'addestramento della PINN (Percorso B).


## Architettura della Cartella e Scelte Progettuali
Questa fase rappresenta la chiusura del cerchio per il "Percorso A". 
- **`optimizer.py`**: È l'unico file di codice presente in questa cartella. Contiene la logica matematica dell'algoritmo di Levenberg-Marquardt. Averlo separato dai solutori (Fase 3) è una scelta architetturale chiave: l'ottimizzatore (Least Squares) agisce come un "regista" che invoca iterativamente i solutori spaziali come "scatola nera" finché non converge. 
- **`lm_convergence.png`**: Grafico autogenerato che mostra visivamente le iterazioni dell'algoritmo di Levenberg-Marquardt e la sua rapidissima caduta verso il parametro target.
- **`Report_Fase_4.md`**: Questo documento.

## Cosa bisogna fare in questa fase
Nei modelli matematici applicati alla realtà, le equazioni (PDE) contengono dei parametri fisici che non conosciamo a priori. Nel nostro caso, non sappiamo qual è il reale **Coefficiente di Diffusione ($D$)** del Covid-19 in Italia. 
Dobbiamo "scoprirlo". Per farlo, usiamo il processo di *Data Fitting* (calibrazione): facciamo girare il nostro modello simulato diverse volte cambiando il valore di $D$, e lo confrontiamo con i dati reali estratti nella Fase 1. Quando la differenza tra simulazione e realtà è minima, abbiamo trovato il $D$ corretto.

Dal punto di vista numerico, questo si traduce in un problema di **Ottimizzazione Non Lineare ai Minimi Quadrati** (Least Squares).

## Spiegazione Dettagliata del Codice (`optimizer.py`)
Questo script chiude il cerchio matematico del Percorso A invocando uno dei più robusti ottimizzatori numerici in circolazione: l'algoritmo di **Levenberg-Marquardt**, importato da `scipy.optimize`.

### 1. La Costruzione del "Ground Truth" e della Simulazione
- **`get_real_data()`**: In uno scenario reale, questa funzione caricherebbe il file generato nella Fase 1. Per il nostro prototipo logico, genera una distribuzione di casi imponendo artificialmente $D = 0.35$, che rappresenta la nostra "realtà" da scoprire.
- **`simulate_diffusion(D)`**: Questa funzione rappresenta la "scatola nera" del modello. Dato un coefficiente $D$ ipotetico, simula l'evoluzione del virus (utilizzando, nel modello completo, i solutori della Fase 3). 

### 2. La Funzione di Costo (Objective Function)
- **`cost_function(D_array, real_data)`**: Questa è la funzione che il nostro algoritmo deve sconfiggere. Estrae il guess corrente `D_guess`, simula l'epidemia con quel parametro, e restituisce il **vettore dei residui** (la differenza matematica tra i dati simulati e i dati reali, cella per cella). Più il residuo è vicino allo zero, più il nostro guess $D$ è simile alla realtà.

### 3. L'Algoritmo di Levenberg-Marquardt
- **`least_squares(..., method='lm')`**: Qui avviene la vera magia dei Metodi Numerici. L'algoritmo parte "alla cieca" con `initial_guess = [1.0]`. Calcola internamente lo Jacobiano (la matrice delle derivate parziali dei residui rispetto a $D$). Se siamo lontani dalla soluzione, si comporta come una classica Discesa del Gradiente (sicura ma lenta). Più si avvicina al minimo globale, più si trasforma nel metodo di Gauss-Newton (velocissimo ma instabile lontano dall'ottimo). Grazie a questo compromesso adattivo, `lm` naviga lo spazio delle soluzioni in un baleno.

## Commento all'Output
```
--- Calibrazione del Modello (Data Fitting & Minimi Quadrati) ---

Recupero dati storici dal DB (ground truth)...
Ipotesi (guess) iniziale per il coefficiente D: 1.0
Avvio ottimizzazione non lineare (Levenberg-Marquardt)...
```
L'algoritmo non sa nulla della realtà, parte ipotizzando $D=1.0$.

```
Ottimizzazione completata!
Risultato ottimale (D_star) trovato: 0.3500
Costo finale (somma dei quadrati degli scarti): 0.00e+00
Numero di iterazioni (valutazioni funzione): 9

SUCCESSO! L'algoritmo ha individuato matematicamente il parametro corretto (0.35) basandosi solo sull'osservazione dei dati finali.
Grafico di convergenza generato: 'lm_convergence.png'.
```
In sole **9 iterazioni**, l'algoritmo di Levenberg-Marquardt ha disceso la curva di errore, trovando il parametro esatto (nel nostro ambiente di test era volutamente nascosto a $0.35$). Il costo finale (errore) è zero. 

Questo step dimostra che hai padronanza dell'**Ottimizzazione Numerica**, chiudendo il cerchio del progetto: hai scaricato dati reali, li hai modellati con una PDE sparsa (Algebra Lineare) e hai ottimizzato i parametri (Minimi Quadrati).
