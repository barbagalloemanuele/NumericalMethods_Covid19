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

## Commento all'Output
*(Il commento effettivo ai risultati verrà scritto qui non appena avremo terminato il run della PINN sul Cluster e generato le immagini definitive)*.
