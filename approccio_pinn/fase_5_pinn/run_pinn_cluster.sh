#!/bin/bash
#SBATCH --job-name=covid_pinn
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --nodelist=gnode10
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2      
#SBATCH --gres=gpu:1 --gres=shard:5000
#SBATCH --output=pinn-training-%j.log

echo "Inizio addestramento intensivo della PINN su GPU Cluster..."

# Run 1: Ottimizzatore Adam (Primo Ordine)
echo "------------------------------------------------------"
echo "Inizio addestramento PINN con ADAM (50,000 epoche)..."
apptainer run --nv /shared/sifs/latest.sif python approccio_pinn/fase_5_pinn/pinn_solver.py --epochs 50000 --env cluster --optimizer adam

# Run 2: Ottimizzatore L-BFGS (Secondo Ordine)
echo "------------------------------------------------------"
echo "Inizio addestramento PINN con L-BFGS (2.500 epoche limite, con stop anticipato per convergenza iterazioni interne)..."
# Per L-BFGS usiamo meno "epoche" perché la closure valuta la funzione 20 volte a step
apptainer run --nv /shared/sifs/latest.sif python approccio_pinn/fase_5_pinn/pinn_solver.py --epochs 50000 --env cluster --optimizer lbfgs

echo "------------------------------------------------------"
echo "Addestramento completato. Controlla la cartella 'modelli_addestrati/cluster' per i risultati."
