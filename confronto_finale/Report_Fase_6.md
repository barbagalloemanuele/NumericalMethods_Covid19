# Report Fase 6: Confronto Finale (Numerico vs PINN)

## Cosa bisogna fare in questa fase
Il progetto si divide in due grandi anime computazionali che risolvono lo stesso identico problema (diffusione di un'epidemia nel tempo e nello spazio):
1. **Percorso A (Approccio Numerico Classico):** PDE alle differenze finite risolte con Matrici Sparse e solutori iterativi accoppiati ai Minimi Quadrati.
2. **Percorso B (Deep Learning / PINN):** Fisica integrata nella funzione di loss di una Rete Neurale Addestrata con Backpropagation.

Questa fase finale raccoglie i risultati di entrambi i percorsi e li mette sotto la lente d'ingrandimento.

## Cosa fa il codice (`compare_models.py` - in sviluppo)
In questa cartella svilupperemo lo script che raccoglie l'output dell'ottimizzatore (Fase 4) e l'output della PINN (Fase 5) per estrarre le metriche chiave.
I grafici che verranno generati (es. `confronto_metodi.png`) metteranno in evidenza:
- **Differenza sul parametro $D$:** Quale metodo si è avvicinato di più al parametro reale partendo dai dati?
- **Tempi di addestramento/esecuzione:** Il calcolo matriciale CPU contro l'ottimizzazione su GPU del Cluster.
- **Robustezza:** Capacità dei modelli di reagire all'assenza di dati (sparce data) o a dati corrotti.

## Commento all'Output
*(Questo file verrà aggiornato con i commenti finali una volta eseguiti sia il percorso A che il percorso B e lanciato lo script di comparazione).*
