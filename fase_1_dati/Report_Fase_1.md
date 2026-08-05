# Report Fase 1: Acquisizione e Trattamento Dati

## Cosa bisogna fare in questa fase
L'obiettivo della Fase 1 è ottenere i dati storici sull'epidemia di COVID-19, filtrarli per il periodo di interesse (la Prima Ondata) e, soprattutto, trasformare la loro rappresentazione geografica in una struttura matematica compatibile con i Metodi Numerici.

Nello specifico dobbiamo:
1. Scaricare il dataset delle province italiane.
2. Isolare le date dal 24 Febbraio 2020 al 31 Maggio 2020.
3. Mappare le coordinate continue (Latitudine e Longitudine) su una griglia discreta (indici X, Y).

## Cosa fa il codice (`data_loader.py`)
Lo script esegue due funzioni principali:
1. `fetch_and_preprocess_data()`: Si collega dinamicamente al repository GitHub della Protezione Civile, scarica il CSV, scarta i dati incompleti (es. "In fase di definizione") e mantiene solo le date della Prima Ondata.
2. `create_spatial_grid()`: Identifica i valori minimi e massimi di latitudine e longitudine in Italia. Quindi, per ogni provincia, calcola una proporzione e le assegna un pixel $(x, y)$ su una griglia predefinita di 50x50.

## Commento all'Output
Quando eseguiamo lo script, otteniamo il seguente output:
```
Downloading data from Protezione Civile...
Data filtered. Shape: (10379, 14)
```
Il dataset originale è enorme; il nostro filtro ha scremato i dati tenendo esattamente le **10.379 registrazioni** valide per la Prima Ondata.

```
Sample of Spatial Mapped Data:
                  data denominazione_provincia  totale_casi  x_idx  y_idx
0  2020-02-24 18:00:00                L'Aquila            0     27     27
1  2020-02-24 18:00:00                  Teramo            0     28     29
...
```
La tabella mostra il successo della mappatura: L'Aquila, ad esempio, è stata assegnata alla riga 27 e colonna 27 della nostra matrice matematica.

```
Unique grid coordinates generated: 105
```
Dei 2500 pixel disponibili sulla griglia 50x50, solo **105 contengono effettivamente una provincia italiana**. I restanti 2395 pixel saranno trattati come "vuoti" o "bordo" nel nostro modello differenziale (ad esempio il mare o i confini esteri). Questi 105 punti saranno le "sorgenti" attive da cui l'equazione di diffusione farà espandere il virus.
