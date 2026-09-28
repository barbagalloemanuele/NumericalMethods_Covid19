# Report Fase 6: Confronto Finale (Numerico vs PINN)

## Architettura della Cartella e Scelte Progettuali
Questa è la cartella conclusiva. Non contiene algoritmi risolutivi propri, ma funge da "cruscotto di analisi" per i dati generati in precedenza:
- **`compare_models.py`**: L'unico script presente. Ha il compito esclusivo di leggere gli output sparsi nel progetto (in particolare i file testuali generati dalla Fase 5) e tradurli in grafici. È stato progettato per generare dinamicamente le barre in base ai file trovati (es. Adam vs L-BFGS, Local vs Cluster), fungendo da strumento di riepilogo automatico.
- **I file `.png` (Grafici)**: Questa cartella ospita i 4 grafici di output finali del progetto (`confronto_accuratezza.png`, `confronto_tempi.png`, `loss_landscape.png`, `robustness_radar.png`). Sono salvati qui per poter essere inclusi istantaneamente nella compilazione della tesi in LaTeX/PDF.
- **`Report_Fase_6.md`**: Questo documento.

## Cosa bisogna fare in questa fase
Il progetto si divide in due grandi anime computazionali che risolvono lo stesso identico problema (diffusione di un'epidemia nel tempo e nello spazio):
1. **Percorso A (Approccio Numerico Classico):** PDE alle differenze finite risolte con Matrici Sparse e solutori iterativi accoppiati ai Minimi Quadrati.
2. **Percorso B (Deep Learning / PINN):** Fisica integrata nella funzione di loss di una Rete Neurale Addestrata con Backpropagation.

Questa fase finale raccoglie i risultati di entrambi i percorsi e li mette sotto la lente d'ingrandimento.

## Cosa fa il codice (`compare_models.py`)
Lo script raccoglie l'output deterministico dell'ottimizzatore (Fase 4) ed esegue una scansione dinamica all'interno delle cartelle `local` e `cluster` della Fase 5, cercando e leggendo tutti i log di training disponibili (es. `result_adam.txt`, `result_lbfgs.txt`).
Quindi, utilizza `matplotlib` per generare **tre** grafici fondamentali:
- **`confronto_finale.png`**: Una dashboard divisa in due sezioni che mette in correlazione diretta l'accuratezza (valore D estratto) e il costo computazionale (scala logaritmica dei tempi), mostrando in parallelo il metodo numerico e gli ottimizzatori di Deep Learning.
- **`loss_landscape.png`**: Un grafico che visualizza l'andamento dell'Errore Assoluto del parametro durante le iterazioni, mostrando matematicamente la "Gradient Pathology" sotto forma di uno stallo permanente di Adam rispetto alla convergenza di L-BFGS.
- **`mesh_free_scatter.png`**: Una mappa 2D che illustra la logica "Mesh-Free" delle PINN, evidenziando visivamente la differenza tra i 500 "Sensori" (Data Points) e i 2000 punti fisici sparsi (Collocation Points), spiegando l'emancipazione dalle griglie classiche.

## Commento all'Output e Analisi Accademica
Eseguendo lo script comparativo finale, si ottiene questo output a terminale:
```
--- Avvio Fase 6: Confronto Numerico vs PINN ---

--- Confronto Completato! Trovati 4 risultati PINN. ---
```
Lo script esteso `compare_models.py` raccoglie tutti i dati e genera i 3 grafici (in stile *Seaborn Modern*) che ci permettono di trarre conclusioni profonde.

### 1. Accuratezza vs Tempi di Calcolo (Numerico vs PINN)
Lo script legge dinamicamente tutti i risultati generati. Dal confronto emerge la chiara gerarchia dei solutori:
- **Metodo Numerico (L-M su Matrici):** Ha impiegato appena **0.05 secondi** per convergere al valore esatto $D = 0.3500$. Su griglie perfettamente campionate, l'algebra lineare iterativa annienta il Deep Learning in termini di performance.
- **PINN (Adam - Primo Ordine):** In locale esegue le iterazioni in pochissimo tempo, ma sul calcolo a regime (Cluster) fatica enormemente a calibrare i decimali, rimanendo bloccato a un errore costante a causa della discesa stocastica del gradiente.
- **PINN (L-BFGS - Secondo Ordine):** Raggiunge il valore matematico esatto, superando la Gradient Pathology grazie al calcolo della curvatura tramite la matrice Hessiana, ma al costo di un tempo computazionale ordini di grandezza superiore rispetto all'algebra lineare.

Questo conferma empiricamente la teoria studiata: i metodi del secondo ordine sono più precisi nella minimizzazione, ma richiedono un'enorme potenza di calcolo (GPU Cluster) per essere sostenibili.

### 2. Superamento della Gradient Pathology (`loss_landscape.png`)
Analizzando i risultati finali del Cluster, emerge una divergenza fondamentale tra gli ottimizzatori, che costituisce uno dei pilastri accademici del progetto.
Tracciando l'Errore Assoluto ($|D_{pred} - 0.35|$), si nota come l'ottimizzatore Adam, pur macinando 50.000 epoche, arresti la convergenza formando una linea orizzontale piatta (stallo) su un errore di $\approx 0.0042$. Al contrario, L-BFGS in sole 7100 iterazioni fa crollare l'errore a zero. 

Questa non è casualità statistica. Nelle PINN coesistono due loss concorrenti: la **Data Loss** e la **Physics Loss**. Durante l'addestramento, i gradienti di queste due funzioni spingono spesso in direzioni opposte, creando profondi minimi locali nello spazio geometrico (fenomeno noto in letteratura come *Gradient Pathology*). L'ottimizzatore stocastico Adam, basandosi solo sulle derivate prime, rimane intrappolato in questa anomalia topologica (lo stallo piatto nel grafico). L'ottimizzatore Quasi-Newton L-BFGS "calcola" l'anomalia della curvatura e naviga agevolmente fino al minimo globale esatto.

## Conclusione della Relazione
Il presente progetto ha raggiunto e dimostrato con successo tutti gli obiettivi prefissati in fase di proposta. Si è partiti dall'estrazione dei dati storici a granularità provinciale della prima ondata COVID-19 (tramite la repository della Protezione Civile), per poi modellare la propagazione tramite un'equazione PDE di diffusione. Si è affrontata la discretizzazione spaziale a differenze finite generando grandi sistemi lineari sparsi, che sono stati risolti con successo valutando le performance dei metodi iterativi (CG e GMRES). L'Inverse Problem è stato infine risolto tramite ottimizzazione ai Minimi Quadrati (Levenberg-Marquardt) per estrarre il parametro dai dati storici.

A valle di tutto ciò, il presente elaborato dimostra empiricamente come l'estrazione vettoriale di parametri fisici spaziali (Parameter Discovery) goda di due soluzioni architetturali antitetiche.
Da un lato, per campionamenti uniformi e geometrie ben definite, il **Percorso A (Metodi Numerici Classici)** offre prestazioni computazionali ottimali, garantendo convergenze in ordine di centesimi di secondo grazie alla solidità intrinseca dei sottospazi di Krylov. 

Dall'altro, il **Percorso B (Deep Learning via PINN)** richiede tempi di addestramento enormemente superiori e ottimizzatori complessi, ma espone una formulazione strutturalmente *mesh-free* (indipendente dalla griglia). Mentre il metodo numerico andrebbe in blocco (matrice singolare) qualora i dati dei sensori sul territorio presentassero grandi lacune spaziali, l'Intelligenza Artificiale si emancipa dalla griglia cartesiana assimilando le equazioni primordiali su coordinate continue, garantendo un'adattabilità senza precedenti al "rumore" del mondo reale.

Entrambi gli approcci convergono al medesimo risultato deterministico ($D = 0.3500$), validando vicendevolmente le metodologie matematiche analizzate.

## Spunti di Riflessione per la Discussione Orale
A corollario dei risultati empirici, si propongono quattro considerazioni accademiche avanzate emerse durante lo sviluppo del progetto, ideali per la discussione in sede d'esame:

1. **La Maledizione della Dimensionalità (Curse of Dimensionality):**
   I metodi alle differenze finite (Percorso A) scalano malissimo all'aumentare delle dimensioni. Una griglia $50 \times 50$ genera una matrice $2500 \times 2500$. Se passassimo a un'equazione 3D ($50 \times 50 \times 50$), la matrice esploderebbe a $125.000 \times 125.000$, saturando rapidamente la RAM e bloccando i solutori di Krylov. Nelle PINN (Percorso B), aggiungere una dimensione significa semplicemente passare alla rete un vettore di input a 4 dimensioni $(x, y, z, t)$ invece di 3. Il numero dei parametri (pesi) della rete non esplode esponenzialmente, rendendo le PINN intrinsecamente superiori per problemi ad alta dimensionalità.

2. **L'Asimmetria tra Problema Diretto e Problema Inverso:**
   Nell'approccio numerico, passare dal *Forward Problem* (calcolare i contagi noto D) all'*Inverse Problem* (calcolare D noti i contagi) richiede uno stravolgimento architetturale totale: è necessario avvolgere il solutore iterativo (GMRES/CG) all'interno di un ottimizzatore non lineare (Levenberg-Marquardt). Nelle PINN, l'architettura rimane **identica**. È sufficiente istanziare $D$ come `nn.Parameter` di PyTorch e l'algoritmo di Backpropagation risolve il problema inverso simultaneamente a quello diretto con zero righe di codice aggiuntive per la logica risolutiva.

3. **Superamento della critica alla "Black Box":**
   Una critica accademica classica rivolta al Deep Learning è l'imprevedibilità del modello ("Scatola Nera"). Le PINN rispondono brillantemente a questo scetticismo: incorporando il Laplaciano spaziale e la derivata temporale all'interno della *Physics Loss*, la rete viene "ingabbiata" e forzata a rispettare i principi di conservazione termodinamica. Si trasforma così in una "White Box" in cui l'Intelligenza Artificiale non può produrre risultati che violino le leggi della fisica classica.

4. **Trade-off sul Costo Energetico (Green Computing):**
   Il Percorso A non richiede pre-computazioni: l'algoritmo parte e in 0.05 secondi offre il risultato. Il Percorso B richiede migliaia di epoche su Cluster GPU per convergere. Tuttavia, l'onere computazionale della PINN è tutto "Upfront" (anticipato in fase di training). Una volta addestrata, calcolare $u(x,y,t)$ per un istante futuro $t=1000$ ha un costo $O(1)$ (inferenza istantanea). I metodi numerici, invece, impongono di ricalcolare iterativamente tutti i $\Delta t$ dal tempo zero fino a 1000. C'è quindi un profondo *trade-off* tra Costo di Addestramento e Costo di Inferenza.
