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

## Spiegazione Dettagliata del Codice (`pinn_solver.py`)
Lo script utilizza `PyTorch` per definire e addestrare una Multi-Layer Perceptron (MLP) profonda che agisce come "solutore universale".

### 1. Architettura della Rete e Parametro Scoperto
- **`nn.Sequential`**: La nostra rete neurale prende 3 numeri in input $(x, y, t)$ e restituisce 1 numero in output: i contagi previsti $u$.
- **`nn.Tanh()`**: A differenza dei classici task di classificazione immagini dove si usa la ReLU, qui la scelta della Tangente Iperbolica è **obbligatoria**. Le derivate seconde della ReLU sono zero ovunque (essendo una spezzata lineare). Dato che dobbiamo calcolare il Laplaciano (che è una derivata seconda) per applicare la fisica, se usassimo la ReLU la *Physics Loss* varrebbe perennemente zero, rompendo l'addestramento.
- **`self.D = nn.Parameter(torch.tensor([1.0]))`**: Questa è l'essenza dell'Inverse Problem. Invece di essere una costante fissa, il coefficiente di diffusione $D$ viene registrato nel Grafo Computazionale di PyTorch. La rete imparerà a correggerlo passo passo, esattamente come corregge i pesi e i bias nascosti.

### 2. Il Motore Fisico (Autograd)
- **`compute_physics_loss`**: Invece di calcolare il Laplaciano facendo "pixel adiacenti meno pixel centrale" (come con le matrici), usiamo **la Derivazione Automatica**. `torch.autograd.grad` viaggia a ritroso nel grafo della rete neurale calcolando la derivata esatta, analitica, di $u$ rispetto a $x, y, t$. Non c'è approssimazione spaziale: le derivate sono pure e continue. Questa funzione calcola la quantità $u_t - D(u_{xx} + u_{yy})$ e punisce la rete proporzionalmente a quanto questo valore si allontana da $0$.

### 3. Addestramento Multi-Obiettivo (Adam vs L-BFGS)
Per garantire una profondità accademica degna del corso di Calcolo Scientifico, lo script non si ferma all'ottimizzatore base, ma implementa un confronto tra metodi del primo e del secondo ordine:
- **`optim.Adam`**: Il re degli ottimizzatori stocastici del primo ordine guida la discesa del gradiente basandosi esclusivamente sulle derivate prime. È veloce ma si arena facilmente nei minimi locali creati dalla "Physics Loss".
- **`optim.LBFGS`**: L'ottimizzatore *Broyden–Fletcher–Goldfarb–Shanno* a memoria limitata. È un metodo del **secondo ordine** che approssima la matrice Hessiana, navigando con incredibile precisione la curvatura dello spazio delle soluzioni. Per usarlo, abbiamo implementato una funzione di `closure()` che ricalcola interamente la loss a ogni sotto-step interno. L-BFGS è il gold standard in letteratura per superare la "Gradient Pathology" delle PINN. 
  *Nota architetturale sul Cluster:* Nello script `run_pinn_cluster.sh`, quando passiamo 50.000 epoche all'algoritmo L-BFGS, il codice Python divide questo numero per 20 (impostando un limite di 2.500 step esterni) poiché la `closure` valuta internamente l'equazione decine di volte. Questo permette all'algoritmo di applicare dinamicamente lo stop anticipato (ad esempio a 7.100 iterazioni totali valutate) in base ai gradienti, garantendo la convergenza senza tempi infiniti.

Durante il loop di training:
  1. Si calcola la **`loss_data`**: Costringe la rete a riprodurre fedelmente i dati reali estratti dal dataset CSV (Fase 1).
  2. Si calcola la **`loss_phys`**: Costringe le predizioni a rispettare le equazioni differenziali.
  3. Si sommano: `loss = loss_data + loss_phys`. 
  L'ottimizzatore calcola il gradiente di questa super-loss rispetto ai pesi e rispetto a $D$, aggiustando il tiro iterativamente.

## Commento all'Output
Quando eseguiamo lo script in locale (con 100 epoche di test), l'output a terminale si presenta così:
```
--- Avvio Training PINN su MPS (Ambiente: local | Ottimizzatore: ADAM) ---
Epoch    1/100 | Loss Totale: 0.1371 | Parametro D: 0.9990
Epoch  100/100 | Loss Totale: 0.0573 | Parametro D: 0.9795

Training completato in 1.07 secondi (ADAM).
==================================================
   Valore D scoperto dalla PINN: 0.9795
   Valore D reale (Protezione Civile): 0.3500
==================================================

Output salvati con successo in:
-> /Users/emanuele/Desktop/.../modelli_addestrati/local/result_adam.txt
```

All'avvio, il parametro $D$ viene inizializzato a $1.0$ (un'ipotesi cieca). 
Con lo scorrere delle epoche di addestramento:
- La **Loss Totale** diminuisce progressivamente (segno che la rete sta trovando un compromesso tra i dati e la fisica).
- Il valore di **$D$ scoperto dalla PINN** cambia gradualmente cercando di avvicinarsi al valore corretto (nella nostra simulazione di test, era nascosto a `0.35`). Tuttavia, come dimostreranno i log sul cluster a 50.000 epoche, a causa della *Gradient Pathology* (il conflitto tra Loss Fisica e Loss Dati), entrambi gli ottimizzatori si fermeranno in un minimo locale sub-ottimale attorno a `0.017`. Questa è un'importante scoperta scientifica del progetto, che attesta i limiti teorici attuali delle PINN standard su problemi non lineari rigidi.
- Al termine dell'esecuzione, i pesi neurali e i file di riepilogo testuali vengono salvati separatamente per ogni ottimizzatore (es. `pinn_model_adam.pth`, `result_lbfgs.txt`), pronti per essere fusi e analizzati nel modulo di confronto (Fase 6).
