# Report Fase 5: Physics-Informed Neural Networks (Percorso B)

## Architettura della Cartella e Scelte Progettuali
Questa è l'unica cartella del **Percorso B (PINN)** ed è l'ambiente più complesso del progetto a livello di file. La sua organizzazione rispecchia il bisogno di eseguire calcoli intensivi sia sul PC locale che sul Cluster universitario:
- **`pinn_solver.py`**: Il cuore pulsante del Deep Learning. A differenza del Percorso A (che usava 3 file per 3 compiti diversi), PyTorch permette di condensare in un solo file la definizione del modello (Equivalente della Fase 2), il calcolo del gradiente e l'ottimizzazione per scoperta parametri (Fasi 3 e 4).
- **`run_pinn_cluster.sh`**: Uno script Bash specifico per l'ambiente SLURM. Siccome il training richiede ore, questo file permette al cluster di prendere in carico il lavoro (usando un container Apptainer con supporto GPU) in modalità asincrona (batch).
- **Cartella `modelli_addestrati`**: Il training di una rete neurale non produce un semplice "numero", ma un set di pesi tensoriali. Questa cartella è stata creata per salvare gli "stati mentali" della rete (i file `.pth`) e separarli rigorosamente per ambiente (`local` vs `cluster`) ed eviatre che le prove rapide su Mac sovrascrivano i pesantissimi risultati del server.
- **`Report_Fase_5.md`**: Questo documento.

## Cosa bisogna fare in questa fase
Entriamo nel **Percorso B (Machine Learning per il Calcolo Scientifico)**. L'obiettivo è risolvere lo stesso problema della Fase 3 e 4, ma senza costruire esplicitamente l'enorme matrice sparsa $A$. 
Invece di discretizzare la fisica su una griglia fissa (Differenze Finite), utilizziamo una **Rete Neurale Artificiale** per imparare la soluzione. 
Il concetto alla base delle *Physics-Informed Neural Networks* (PINN) è brillante: istruiamo la rete neurale a minimizzare un errore (Loss) che non tiene conto solo dei dati misurati, ma che la "punisce" anche se produce output che violano le leggi della fisica (nel nostro caso, l'equazione differenziale della diffusione).

## Spiegazione del Codice Riga per Riga (`pinn_solver.py`)
Lo script utilizza `PyTorch` per definire e addestrare una Multi-Layer Perceptron (MLP) profonda.

- **`Righe 20-28: self.net = nn.Sequential(...)`**
  Costruiamo l'architettura della rete. Essa accetta 3 feature in input (coordinate spaziali $x, y$ e temporale $t$) e restituisce uno scalare $u$ (contagi stimati). Include 3 hidden layer lineari.
- **`Righe 22, 24, 26: nn.Tanh()`**
  In ambito PINN, la scelta della funzione di attivazione non può ricadere sulla `ReLU` (Rectified Linear Unit), poiché le sue derivate seconde sono identicamente nulle ovunque (eccetto nell'origine dove non è differenziabile). Il calcolo del Laplaciano spaziale della PDE richiede derivate seconde non nulle per poter addestrare la rete. La Tangente Iperbolica (`Tanh`) fornisce le proprietà di differenziabilità globale e regolarità richieste per il calcolo tramite autograd.
- **`Riga 33: self.D = nn.Parameter(torch.tensor([1.0]))`**
  Implementazione computazionale dell'Inverse Problem. Il coefficiente di diffusione $D$, fisicamente incognito, è registrato nell'albero dei parametri ottimizzabili di PyTorch (`nn.Parameter`). Durante la backpropagation, il gradiente della loss rispetto a $D$ verrà calcolato, permettendo l'aggiornamento iterativo del parametro.
- **`Righe 52-62: torch.autograd.grad(..., create_graph=True)`**
  Applicazione della Derivazione Automatica (Autograd). Questa API calcola la derivata esatta, in forma analitica, dell'output della rete neurale rispetto agli input spaziali e temporali. Il flag `create_graph=True` mantiene in memoria il grafo computazionale delle derivate prime, condizione obbligatoria per poter applicare un secondo operatore derivativo spaziale e ottenere $u_{xx}$ e $u_{yy}$.
- **`Riga 65: residual = u_t - model.D * (u_xx + u_yy)`**
  Valutazione dell'equazione differenziale alle derivate parziali (PDE). Il termine `residual` rappresenta lo scostamento puntuale della rete neurale rispetto alla dinamica della diffusione continua.
- **`Riga 83: D_real = 0.35`**
  Inizializzazione del parametro "ground truth" per il benchmark dell'ottimizzatore.
- **`Riga 86: u = torch.exp(-D_real * t) * torch.sin(np.pi * x) * torch.sin(np.pi * y)`**
  Questa riga calcola una formula matematica nota, che è la soluzione esatta dell'equazione di diffusione. Siccome in questo script di test non stiamo ancora caricando i file CSV reali, usiamo questa formula per generare dei "dati perfetti". La rete neurale non conosce la formula, riceve in input solo i numeri finali e deve analizzarli per imparare a simulare il contagio.
- **`Righe 101-103: optimizer = optim.LBFGS(...)`**
  Setup condizionale dell'ottimizzatore. L-BFGS esegue internamente iterazioni multiple (fino a `max_iter=20`) per ogni singola chiamata di `optimizer.step()`. Di conseguenza, per standardizzare il computo globale delle epoche, riduciamo il ciclo for esterno dividendolo per 20 (`epochs = epochs // 20`).
- **`Righe 107-113: Data Points vs Physics Points`**
  La logica architetturale delle PINN prevede l'istanziazione di due distinti domini vettoriali:
  1. `Data Points` (500 elementi): Coordinate per le quali è noto a priori il vettore target `u_data`. Impiegati nel calcolo dell'errore di Mean Squared Error (Data Loss).
  2. `Collocation Points` (2000 elementi): Coordinate estratte casualmente nel dominio che *non* possiedono un target (`_`). Impiegati esclusivamente per minimizzare il residuo dell'equazione fisica spaziale (Physics Loss) sfruttando l'astrazione continua fornita dalla rete.
- **`Righe 123-130: Loop di Addestramento Adam`**
  Pipeline standard del forward-backward pass. Si azzerano i gradienti precedenti (`optimizer.zero_grad()`), si genera un operatore differenziabile globale (`loss = loss_data + loss_phys`), lo si propaga per ricavare i gradienti sui nodi (`loss.backward()`) e si aggiornano i pesi vettoriali (`optimizer.step()`).
- **`Riga 136: def closure():`**
  Per comprendere questa funzione, bisogna capire la logica meccanica di L-BFGS. Quando Adam esegue un aggiornamento dei pesi, valuta la Loss esattamente 1 volta. Quando L-BFGS deve aggiornare i pesi, calcola la direzione di discesa, ma prima di applicare l'aggiornamento deve decidere l'ampiezza perfetta del passo (*Line Search*). Per farlo, fa molteplici "tentativi" provvisori lungo quella direzione, ricalcolando la Loss ad ogni tentativo. 
  Affinché l'algoritmo PyTorch possa ricalcolare la Loss autonomamente in background durante questi tentativi, ha bisogno di sapere *come* si calcola. La `closure()` è semplicemente un pacchetto di istruzioni (che contiene il forward-pass e il calcolo dei gradienti) che noi cediamo a L-BFGS, permettendogli di auto-eseguirlo decine di volte all'interno di un singolo step.

## Q&A per la Presentazione (Approfondimenti Teorici)

**1. Qual è il significato matematico della formula alla riga 86?**
L'equazione $u(x,y,t) = e^{-D t} \sin(\pi x) \sin(\pi y)$ è un modello matematico (Modello Giocattolo / *Toy Model*) che simula una dinamica di diffusione spaziale. Analizziamola matematicamente:
- **La componente spaziale ($\sin(\pi x) \sin(\pi y)$)**: Su un dominio di coordinate $x \in [0,1]$ e $y \in [0,1]$, questa moltiplicazione di seni crea una superficie a forma di "cupola" o "campana". Ai bordi (es. $x=0$ o $y=1$) il seno vale zero, simulando una condizione al contorno di Dirichlet omogenea (fuori dalla regione non c'è contagio). Al centro ($x=0.5, y=0.5$) raggiunge il picco massimo.
- **La componente temporale ($e^{-D t}$)**: L'esponenziale negativo impone un decadimento nel tempo. Man mano che il tempo $t$ avanza, l'intera "cupola" di contagi si sgonfia verso lo zero in modo proporzionale al parametro di diffusione $D$. 
Sintetizzando, questa equazione descrive un focolaio epidemico isolato al centro dell'Italia che si dissipa col tempo.

**2. Qual è lo scopo matematico dei 500 Data Points e dei 2000 Collocation Points?**
Questo è il fondamento architetturale delle PINN, che fonde l'apprendimento supervisionato a quello non supervisionato:
- **I 500 Data Points (Supervised Learning):** Sono coordinate spaziotemporali $(x,y,t)$ associate a un valore noto (target) di contagi $u$. Rappresentano i nostri "sensori" sul territorio. La rete calcola la *Data Loss* minimizzando il Mean Squared Error tra la sua previsione e questi 500 valori reali. Se avessimo solo questi punti, la rete neurale (che è un approssimatore universale) "sovradatterebbe" (overfitting), creando una funzione matematica folle e seghettata pur di passare esattamente per i 500 punti.
- **I 2000 Collocation Points (Physics Regularization):** Sono punti sparsi in cui *non* conosciamo i contagi reali. Su questi punti, la rete neurale viene obbligata a rispettare l'equazione differenziale (Loss Fisica). Hanno un ruolo di "regolarizzatore matematico". Costringono la funzione della rete neurale a essere liscia, fisicamente coerente e a comportarsi come una diffusione spaziale anche nelle vaste zone vuote tra un sensore e l'altro. Ne usiamo 2000 (il quadruplo dei dati) proprio per "ingabbiare" fisicamente l'intero dominio spaziale, impedendo all'AI di inventare andamenti termodinamicamente impossibili nelle zone non coperte da sensori.

**3. L-BFGS e differenze algoritmiche con Adam**
Adam (Adaptive Moment Estimation) è un ottimizzatore del **primo ordine**: aggiorna i pesi neurali computando esclusivamente stime adattive del gradiente della funzione di costo e dei suoi momenti (varianza). 
L-BFGS (Broyden–Fletcher–Goldfarb–Shanno) è un metodo del **secondo ordine** appartenente alla famiglia Quasi-Newton. Calcola l'aggiornamento dei parametri approssimando la matrice Hessiana inversa. Poiché dispone delle informazioni relative alle derivate seconde (la curvatura spaziale della loss), converge verso minimi complessi con un numero marcatamente inferiore di step globali rispetto ai metodi stocastici del primo ordine.

**4. Cos'è la peculiarità "Limited-memory" di L-BFGS?**
Calcolare, allocare e invertire analiticamente un'autentica matrice Hessiana per una rete neurale ha un costo computazionale e spaziale nell'ordine di $O(N^2)$ (con $N$ pari ai pesi totali della rete), superando i limiti fisici della RAM. Il prefisso "Limited-memory" indica la soluzione algoritmica di non istanziare interamente la matrice, stimando invece la sua inversa salvando vettorialmente soltanto gli ultimi gradienti processati (parametro `history_size=100`).

**5. Perché è preferito l'uso di L-BFGS per le reti PINN?**
Il paesaggio della funzione di costo (Loss landscape) delle PINN è intrinsecamente rigido ("stiff") poiché risulta dalla perturbazione asimmetrica tra l'errore di minimizzazione puntuale dei dati reali e l'errore differenziale dell'equazione della fisica (Gradient Pathology). Gli ottimizzatori stocastici del primo ordine arrestano prematuramente la convergenza impigliandosi in minimi locali sub-ottimali. L-BFGS modula iterativamente il proprio aggiornamento tramite la curvatura stimata, risultando lo standard in letteratura per ovviare alla Gradient Pathology.

## Analisi Comparativa delle Esecuzioni (Locale vs Cluster)

Il codice è stato concepito per un duplice setup di build, sfruttando il modulo os per separare log e dump binari (`.pth`).

### Esecuzione di Debug (Locale)
L'esecuzione sull'ambiente `local` è impiegata a scopo di verifica compilazione del grafo differenziale per 100 epoche nominali. I risultati testuali da `modelli_addestrati/local/` evidenziano:
- **Adam (100 epoche, 1.07s)**: Esegue 100 cicli classici. Loss $5.73 \times 10^{-2}$ | Parametro scoperto: `0.4501`
- **L-BFGS (105 valutazioni effettive, 5.93s)**: Nel file Python, avevamo richiesto che per L-BFGS il numero di epoche venisse diviso per 20 (`100 // 20 = 5`). Il ciclo `for` è durato quindi esattamente 5 passi. Perché leggiamo 105? Perché il nostro contatore globale non contava i cicli `for`, ma contava le esecuzioni della `closure()`. Durante ciascuno dei 5 cicli esterni, L-BFGS ha valutato internamente l'errore circa 21 volte (cercando la lunghezza di passo perfetta), totalizzando 105 esecuzioni. Grazie a queste prove interne, ha indovinato il parametro `0.3499` quasi istantaneamente.

### Esecuzione HPC (Cluster UNICT)
Tramite lo script bash `run_pinn_cluster.sh`, i layer computazionali sono stati eseguiti in job asincroni via SLURM, per un tetto massimo di $50.000$ epoche nominali (risoluzione dell'Inverse Problem). Dai file in `modelli_addestrati/cluster/` deduciamo la metrica di calibrazione conclusiva:
1. **Risultato Adam (`result_adam.txt`)**: Macinando 50.000 epoche in 263s, Adam riduce la Loss a $1.88 \times 10^{-6}$, attestando la stima $D$ a **0.3542**. Evidente dimostrazione della Gradient Pathology nel mancato perfezionamento dei decimali finali in fase di minimo locale ristretto.
2. **Risultato L-BFGS (`result_lbfgs.txt`)**: Registra la precisione teorica assoluta. Grazie al calcolo Hessiano, in sole 7100 iterazioni totalizzate per line-search, arresta il loop interno anticipatamente (a metà tempistica rispetto ad Adam, 131s) calibrando il parametro rigorosamente al ground truth analitico: **0.3500**, con una Loss di $4.41 \times 10^{-6}$.

## Conclusione della Fase 5
Lo scopo prefissato del **Percorso B** era verificare se fosse possibile modellare e risolvere il problema epidemiologico spaziale abbandonando totalmente l'infrastruttura matematica classica.
La conclusione è un successo ingegneristico: **senza costruire alcuna griglia di nodi, senza allocare matrici sparse e senza impiegare alcun solutore iterativo di Krylov**, una Rete Neurale Artificiale è stata in grado di interiorizzare le leggi della diffusione termodinamica. 
Sfruttando unicamente la derivazione automatica (Autograd) e un ottimizzatore del secondo ordine (L-BFGS), la PINN ha estratto con precisione assoluta il coefficiente fisico ($D=0.3500$) dalla mera osservazione di punti sparsi. Il modello addestrato (salvato come tensore `.pth`) costituisce ora l'alter-ego in Deep Learning dei Metodi Numerici visti nella Fase 4.
