#!/bin/bash
#SBATCH --job-name=covid_pinn
#SBATCH --account=SostituisciConTuoAccount
#SBATCH --partition=SostituisciConTuaCoda
#SBATCH --qos=gpu-medium
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2      
#SBATCH --gres=gpu:1 --gres=shard:8000
#SBATCH --output=logs/pinn-training-%j.log

echo "Inizio addestramento intensivo della PINN su GPU Cluster..."

# Eseguiamo il codice Python tramite l'immagine Apptainer con 50.000 epoche
apptainer run --nv /shared/sifs/latest.sif python approccio_pinn/fase_5_pinn/pinn_solver.py --epochs 50000 --env cluster

echo "Addestramento completato. Controlla la cartella 'modelli_addestrati' per il file .pth"
