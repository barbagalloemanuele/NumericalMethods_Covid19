# Report Fase 1: Acquisizione e Trattamento Dati

## Architettura della Cartella e Scelte Progettuali
Questa cartella contiene strettamente il necessario per l'estrazione e preparazione dei dati, isolando il data-engineering dal calcolo matematico successivo:
- **`data_loader.py`**: È l'unico script Python. La scelta di non salvare file `.csv` giganteschi nella cartella, ma di scaricarli "on-the-fly" dalla Protezione Civile tramite Pandas, è stata fatta per mantenere il repository leggero ed evitare conflitti di versione o dati obsoleti.
- **`covid_raw_data.png`**: Grafico autogenerato dallo script che mostra visivamente l'andamento puro dei contagi in Italia nel periodo studiato, utile per l'introduzione della tesi.
- **`Report_Fase_1.md`**: Questo documento.

## Cosa bisogna fare in questa fase
L'obiettivo della Fase 1 è ottenere i dati storici sull'epidemia di COVID-19, filtrarli per il periodo di interesse (la Prima Ondata) e, soprattutto, trasformare la loro rappresentazione geografica in una struttura matematica compatibile con i Metodi Numerici.

Nello specifico dobbiamo:
1. Scaricare il dataset delle province italiane.
2. Isolare le date dal 24 Febbraio 2020 al 31 Maggio 2020.
3. Mappare le coordinate continue (Latitudine e Longitudine) su una griglia discreta (indici X, Y).

## Spiegazione Dettagliata del Codice (`data_loader.py`)
Lo script ruota attorno a due funzioni core scritte in `pandas`:

### 1. `fetch_and_preprocess_data()`
Questo blocco si occupa dell'ingestion e del filtraggio temporale:
- **`pd.read_csv(url)`**: Scarica "on the fly" l'intero dataset storico del dipartimento di Protezione Civile bypassando la necessità di salvare giga di file sul PC.
- **`dt.tz_localize(None)`**: Questo è un passaggio critico. I dati scaricati possiedono un fuso orario (Timezone UTC). Rimuovendo il fuso orario permettiamo a Pandas di fare comparazioni temporali assolute (es. "dal 24 Febbraio al 31 Maggio") senza incorrere in crash legati ai cambi dell'ora legale.
- **`df.dropna(...)`**: Molte province nel dataset sono etichettate come "In fase di definizione", il che significa che non hanno coordinate geografiche (Lat/Long = 0). Siccome la nostra PDE necessita dello spazio fisico, queste righe vengono scartate.
- **Aggregazione e Plot (`matplotlib`)**: Prima di restituire i dati provinciali per la griglia spaziale, il codice raggruppa tutti i dati per data (`groupby('data')`) e somma i casi, così da poter salvare il grafico esplorativo iniziale `covid_raw_data.png`.

### 2. `create_spatial_grid()`
Questo blocco effettua una normalizzazione min-max per mappare coordinate spaziali reali su una griglia matematica (la matrice che useremo nelle fasi successive):
- **`min_lat, max_lat`**: Vengono estratti i bordi geografici estremi dell'Italia.
- **Interpolazione Lineare `((val - min) / (max - min)) * grid_size`**: Questa formula geniale prende una coordinata (es. Longitudine 12.3) e la ri-mappa in una percentuale (es. 40%). Moltiplicandola per la dimensione della nostra griglia (50x50), otteniamo esattamente l'indice $X, Y$ (es. pixel 20) all'interno del nostro spazio vettoriale in cui collocare i casi di contagio di quella specifica provincia.

## Commento Esplicativo all'Output
Quando eseguiamo lo script, otteniamo il seguente output:
```
Downloading data from Protezione Civile...
Grafico dei dati storici generato: 'covid_raw_data.png'.
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
