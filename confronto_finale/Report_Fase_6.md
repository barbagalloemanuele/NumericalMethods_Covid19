# Report Fase 6: Confronto Finale (Numerico vs PINN)

## Cosa bisogna fare in questa fase
Il progetto si divide in due grandi anime computazionali che risolvono lo stesso identico problema (diffusione di un'epidemia nel tempo e nello spazio):
1. **Percorso A (Approccio Numerico Classico):** PDE alle differenze finite risolte con Matrici Sparse e solutori iterativi accoppiati ai Minimi Quadrati.
2. **Percorso B (Deep Learning / PINN):** Fisica integrata nella funzione di loss di una Rete Neurale Addestrata con Backpropagation.

Questa fase finale raccoglie i risultati di entrambi i percorsi e li mette sotto la lente d'ingrandimento.

## Cosa fa il codice (`compare_models.py`)
Lo script raccoglie l'output deterministico dell'ottimizzatore (Fase 4) e legge il file `result.txt` generato dalla PINN (Fase 5) nella cartella `local` o `cluster`.
Quindi, utilizza `matplotlib` per generare due grafici fondamentali:
- **`confronto_accuratezza.png`**: Un grafico a barre che mostra affiancati il valore reale di $D$ (ground truth), il valore calcolato dal metodo numerico (L-M) e il valore scoperto dalla Rete Neurale.
- **`confronto_tempi.png`**: Un grafico a barre in scala logaritmica che mette a confronto la sbalorditiva differenza di tempo (secondi vs ore/minuti) tra l'ottimizzazione pura di matrici e la Backpropagation neurale.

## Commento all'Output e Analisi Accademica
Lo script esteso `compare_models.py` raccoglie tutti i dati e genera 4 grafici fondamentali (in stile *Seaborn Modern*) che ci permettono di trarre conclusioni molto più profonde della semplice velocità.

### 1. Accuratezza vs Tempi di Calcolo
- **Metodo Numerico (L-M su Matrici):** Ha impiegato appena **0.05 secondi** per convergere al valore esatto $D = 0.3500$.
- **PINN (Deep Learning):** Ha richiesto **246 secondi** (su GPU L40S) per fermarsi a un sub-ottimo di $D = 0.0178$.
Questo dimostra che, su griglie perfettamente campionate, l'algebra lineare iterativa annienta il Deep Learning per i task di *Inverse Parameter Discovery*.

### 2. Il Conflitto dei Gradienti (`loss_landscape.png`)
Perché la PINN ha fallito l'accuratezza? Il grafico *Loss Landscape* lo spiega visivamente: in una PINN coesistono la **Data Loss** e la **Physics Loss**. Durante l'addestramento, l'ottimizzatore Adam cerca di abbassarle entrambe, ma i loro gradienti spesso spingono in direzioni opposte (Gradient Pathology). Questo causa lo stallo della rete in un minimo locale sub-ottimale. Per risolvere questo problema servirebbero milioni di epoche, architetture a pesi dinamici (Self-Adaptive Loss Weights) o ottimizzatori del secondo ordine (L-BFGS).

### 3. Compromessi Architetturali (`robustness_radar.png`)
Nonostante la sconfitta sui tempi e sulla precisione teorica, il **Radar Chart** rivela i veri superpoteri delle PINN:
- **Robustezza ai Dati Sparsi (Mesh-Free):** I solutori numerici (Percorso A) necessitano di una griglia ininterrotta per calcolare il Laplaciano. Se mancassero i dati di una regione centrale d'Italia, l'algoritmo numerico crollerebbe o andrebbe fuori memoria nel tentativo di bypassarla. Le PINN, operando su coordinate pure $(x,y,t)$ senza griglie fisiche, possono imparare l'equazione di diffusione anche se abbiamo solo il 5% dei dati reali sparsi a macchia di leopardo!
- **Estrapolazione Continua (Continuous Time):** I metodi numerici devono calcolare ogni singolo istante di tempo ($\Delta t$) per poter predire il futuro. La PINN impara una funzione surrogata continua $u(x,y,t)$. Una volta addestrata (anche mettendoci ore), l'inferenza è istantanea per qualsiasi tempo $t=1000$ senza dover iterare i 999 step precedenti.

## Conclusione della Relazione
Per il *Parameter Discovery* epidemico su dataset ben formati e strutturati a griglia, il **Percorso A (Metodi Numerici Classici)** rimane lo stato dell'arte in termini di velocità e precisione. 
Tuttavia, il **Percorso B (PINN)** offre una flessibilità ineguagliabile che lo rende l'unica strada percorribile quando si lavora nel mondo reale con sensori guasti, dati frammentati e geometrie spaziali complesse in cui è impossibile costruire una matrice sparsa.
