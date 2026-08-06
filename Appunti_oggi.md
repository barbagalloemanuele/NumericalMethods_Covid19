# Appunti del Progetto - Sessione Odierna (Fase di Riorganizzazione e PINN)

## Cosa abbiamo fatto oggi:
La giornata di oggi è stata cruciale per l'allineamento del progetto alle nuove direttive del Prof. Boscarino (Confronto tra Metodi Numerici e Reti Neurali).

1. **Riorganizzazione Totale:** 
   - Abbiamo pulito l'intera struttura del progetto passando da un workflow lineare a un **Bivio Analitico**: `Modulo Base` -> `Percorso A` (Numerico) o `Percorso B` (PINN) -> `Confronto Finale`.
   - Abbiamo rinominato e riordinato tutte le cartelle (`approccio_numerico`, `approccio_pinn`, `confronto_finale`) affinché il prof possa leggere un'infrastruttura pulita.
   
2. **Aggiornamento Documentazione (Report e SKILL):**
   - Abbiamo aggiornato tutti i `Report_Fase_X.md` pre-esistenti inserendo una nota introduttiva che li inquadra all'interno del *Percorso A*.
   - Abbiamo spostato il vecchio script dei grafici `visualize.py` al suo vero posto (`fase_3_solutori`), dove aiuterà a mostrare il crollo del residuo per CG e GMRES.
   - Abbiamo creato i file `.agents/skills/.../SKILL.md` per fissare le regole architettoniche (specie la gestione del cluster e la separazione dei due percorsi).

3. **Ingegnerizzazione Percorso B (Physics-Informed Neural Networks):**
   - Abbiamo scritto l'intero codice della rete neurale `pinn_solver.py` in PyTorch, implementando un'attivazione *Tanh* e una Loss Multi-Obiettivo (*Data* + *Physics Loss* calcolata via Autograd).
   - Abbiamo testato l'addestramento in locale sul Mac (sfruttando l'accelerazione MPS) usando 1000 epoche. 
   - **Scoperta scientifica:** Come previsto, 1000 epoche su CPU/MPS non sono sufficienti per permettere a una PINN di scoprire il parametro `D` esatto (si è fermata a 0.0086 invece di 0.35). Questo legittima l'utilizzo dei metodi numerici "classici" (che ci hanno messo solo 9 iterazioni) e dimostra la necessità di usare il cluster per addestramenti più feroci!
   - Abbiamo strutturato l'output in modo che il codice riconosca se sta girando su Mac (`--env local`) o su Server (`--env cluster`), creando cartelle logiche per salvare il modello (`.pth`) e un comodo file testuale `result.txt`.

4. **Sottomissione su Cluster GPU UNICT:**
   - Abbiamo configurato con successo il file di lancio `run_pinn_cluster.sh` integrando i parametri *SLURM* corretti (Account, Partizione e QoS).
   - *Attualmente il job è in esecuzione sulle GPU L40S/V100 del cluster con ben 50.000 epoche!*

5. **Preparazione per il Finale (Fase 6):**
   - Abbiamo scritto a priori lo script Python `compare_models.py` (nella cartella del confronto finale).
   - Non appena l'addestramento sul cluster sarà finito e sincronizzato sul Mac (tramite `rsync`), questo script leggerà in automatico i `result.txt` e genererà i due bellissimi grafici conclusivi: il confronto di **accuratezza del parametro D** (Bar Chart) e il confronto sui **tempi di calcolo** (su scala logaritmica).

## Prossimi Passi (per chiudere il progetto):
1. Aspettare la fine del job sul cluster.
2. Sincronizzare da remoto (`rsync`).
3. Lanciare `python confronto_finale/compare_models.py`.
4. Inserire i risultati (Plot e Dati) all'interno del PDF/Relazione Finale e inviare al Prof!
