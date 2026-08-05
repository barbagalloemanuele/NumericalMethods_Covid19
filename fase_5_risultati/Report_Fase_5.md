# Report Fase 5: Visualizzazione e Risultati (Deliverables)

## Cosa bisogna fare in questa fase
Il codice numerico nudo e crudo produce solo numeri. Per la relazione finale e per dimostrare ai professori la bontà del lavoro, è necessario generare dei grafici (*Deliverables*) che permettano di visualizzare immediatamente i risultati delle simulazioni numeriche, come ad esempio le performance dei metodi iterativi studiati.

## Cosa fa il codice (`visualize.py`)
Lo script utilizza la libreria `matplotlib` per creare un grafico in scala semilogaritmica sull'asse Y (`plt.semilogy`). Questa scala è lo standard accademico in Analisi Numerica per mostrare il decadimento esponenziale dell'errore (il residuo relativo) iterazione dopo iterazione.
Il grafico traccia i due metodi a confronto: Gradiente Coniugato (blu, linea continua) e GMRES (rosso, linea tratteggiata), mostrando chiaramente come si avvicinano alla retta nera puntata che rappresenta la tolleranza limite fissata a $10^{-6}$.

## Commento all'Output
Lo script non mostra il grafico a schermo bloccando il terminale, ma salva direttamente un file immagine ad alta risoluzione (300 dpi) pronto per la stampa, chiamato `convergenza_solutori.png`, nella cartella principale del progetto.
```
--- Fase 5: Generazione Grafici Finali (Deliverables) ---
Grafico di convergenza salvato con successo come 'convergenza_solutori.png' nella cartella corrente.
```
Questo file PNG può essere allegato direttamente in un documento PDF/LaTeX per la relazione finale del progetto.
