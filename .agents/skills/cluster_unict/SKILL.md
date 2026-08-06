---
name: Utilizzo Cluster UNICT (SLURM & Apptainer)
description: Istruzioni per l'accesso, la sincronizzazione del codice e l'esecuzione di job GPU sul cluster dell'università (gcluster.dmi.unict.it).
---

# Istruzioni per l'uso del Cluster UNICT

Questo skill definisce come interfacciarsi con il cluster GPU dell'università.

## 1. Connessione e Trasferimento File
Il cluster non va usato per sviluppare codice direttamente, ma solo per l'esecuzione intensiva.
- **Sviluppo:** Avviene in locale (sul Mac).
- **Sincronizzazione:** Per inviare il codice al cluster si usa `rsync` dalla macchina locale:
  ```bash
  rsync -avz --progress ./ username@gcluster.dmi.unict.it:~/progetto_metodi_numerici/ --exclude .venv --exclude __pycache__ --exclude .git
  ```
- **Recupero Risultati:** 
  ```bash
  rsync -avz username@gcluster.dmi.unict.it:~/progetto_metodi_numerici/risultati/ ./risultati_dal_cluster/
  ```

## 2. Esecuzione dei Job (SLURM + Apptainer)
Il cluster usa SLURM per la gestione delle code e Apptainer per gli ambienti virtuali isolati.
- **MAI eseguire codice pesante sul nodo di login.**
- I job si inviano tramite script `.sh` da sottomettere con `sbatch`.

### Struttura base di uno script sbatch (es. run_pinn.sh)
```bash
#!/bin/bash
#SBATCH --account=NOME_ACCOUNT_ASSEGNATO
#SBATCH --partition=NOME_CODA_ASSEGNATA
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2      
#SBATCH --gres=gpu:1 --gres=shard:5000
#SBATCH --output=logs/job-%j.log

# Esecuzione del codice PyTorch tramite il container preconfigurato del cluster
apptainer run --nv /shared/sifs/latest.sif python approccio_pinn/fase_5_pinn/pinn_solver.py
```

## 3. Gestione e Monitoraggio
- Sottomettere il job: `sbatch run_pinn.sh`
- Controllare i propri job: `squeue --me`
- Annullare un job: `scancel <JOB_ID>`
- Leggere l'output in tempo reale: `tail -f logs/job-<JOB_ID>.log`
- Controllare limiti disco: `quota -s`
