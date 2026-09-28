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

### 1. Il Download Dinamico e la Pulizia (`fetch_and_preprocess_data()`)
- **`pd.read_csv(url)`**: Scarica "on the fly" l'intero dataset storico della Protezione Civile. Invece di salvare un CSV sul PC (col rischio che diventi vecchio), ci colleghiamo in tempo reale, rendendo il codice portatile.
- **La correzione del Timezone (`dt.tz_localize(None)`)**: I dati governativi includono il fuso orario UTC. Se non rimuovessimo il timezone, i filtri di Pandas (che usano date "naive") crasherebbero. Questo previene gli errori legati all'ora legale.
- **Il filtro Temporale**: Ritagliamo solo la Prima Ondata (dal 24 Febbraio al 31 Maggio 2020) per avere un'epidemia pura, senza lockdown mirati, varianti o vaccini che distruggerebbero il modello differenziale.
- **Il pericolo dell'Oceano Atlantico (`df.dropna(...)` e `lat/long != 0`)**: Quando la Protezione Civile non sa dove allocare dei contagi (es. "In fase di definizione"), assegna Lat=0 e Long=0. Se non eliminassimo questi dati sporchi, il modello collocherebbe migliaia di infetti nel Golfo di Guinea, corrompendo tutta la matematica spaziale.

### 2. Il Grid Mapping (`create_spatial_grid()`)
Questa funzione trasforma la mappa d'Italia da uno spazio continuo a una scacchiera discreta (matrice 50x50):
- **Perché proprio 50x50? (Scelta Architetturale)**: Non bisogna dare per scontata questa dimensione. Se avessimo scelto una griglia molto piccola (es. 10x10), le coordinate si sarebbero schiacciate troppo, accorpando province molto distanti in una singola casella e distruggendo la risoluzione spaziale del contagio. Al contrario, se avessimo scelto una griglia ad altissima risoluzione (es. 1000x1000), la matrice matematica risultante nella Fase 2 avrebbe avuto dimensioni 1 milione x 1 milione. Questo avrebbe consumato inutilmente un'immensa quantità di RAM senza alcun reale beneficio, dal momento che i capoluoghi di provincia italiani sono sempre e solo 105. Una griglia 50x50 (che produce un sistema gestibile da 2500 celle totali) rappresenta il perfetto compromesso accademico tra fedeltà geografica e leggerezza computazionale.
- **Interpolazione Lineare `((val - min) / (max - min)) * grid_size`**: Usa la Normalizzazione Min-Max. Prende la coordinata GPS esatta di una provincia, ne calcola la distanza percentuale dai bordi estremi d'Italia, e la moltiplica per 49. Il comando `.astype(int)` tronca i decimali. Così facendo, le coordinate GPS complesse diventano un semplice indice $(X, Y)$ (ad esempio, Milano diventa riga 12, colonna 38).

## Commento Esplicativo all'Output
Quando eseguiamo lo script, otteniamo il seguente output a terminale:
```
Downloading data from Protezione Civile...
Grafico dei dati storici generato: 'covid_raw_data.png'.
Data filtered. Shape: (10379, 14)
```
Il dataset grezzo originale è gigantesco (copre 4 anni di pandemia); il nostro filtro temporale lo ha scremato isolando esattamente le **10.379 registrazioni** valide per la Prima Ondata. Questo set di dati rappresenta il nostro *Ground Truth*, ovvero la "verità assoluta" su cui calibreremo i parametri dell'equazione differenziale nelle fasi 4 e 5.

```
Sample of Spatial Mapped Data:
                  data denominazione_provincia  totale_casi  x_idx  y_idx
0  2020-02-24 18:00:00                L'Aquila            0     27     27
1  2020-02-24 18:00:00                  Teramo            0     28     29
...
```
La stampa a schermo dimostra il successo del *Grid Mapping*: l'algoritmo ha convertito le complesse coordinate GPS di L'Aquila, assegnandole alla riga 27 e colonna 27 della nostra futura matrice spaziale. 

```
Unique grid coordinates generated: 105
```
Un dettaglio ingegneristico di cruciale importanza: le province italiane ufficiali estrapolate dal dataset sono 107. Tuttavia, l'algoritmo restituisce **105 coordinate spaziali attive**. Questo accade perché, alla risoluzione della nostra griglia 50x50, due specifiche coppie di province (*Firenze-Prato* e *Messina-Reggio Calabria*) sono geograficamente così vicine da collassare all'interno dello stesso identico pixel matematico. 
Questa non è un'anomalia, ma una precisa scelta architetturale (Trade-off):
1. Aumentare la griglia (es. 100x100) per separare queste province farebbe esplodere la matrice del Laplaciano a 10.000 x 10.000 (100 milioni di celle), saturando la RAM di calcolo.
2. Dal punto di vista del modello epidemico macroscopico, province distanti meno di 20km o separate da uno stretto marittimo, e unite da un pendolarismo massiccio, agiscono dinamicamente come **un unico focolaio virale**. Fonderle in una sola "sorgente" PDE è non solo computazionalmente efficiente, ma anche fisicamente realistico.

In sintesi, delle 2500 celle disponibili sulla nostra griglia, solo **105 fungono da Sorgenti Attive** (pari al 4.2% del dominio). I restanti 2395 pixel saranno matematicamente trattati come "vuoti" o "bordo" nel modello differenziale (rappresentando il mare o i confini geografici esteri).
