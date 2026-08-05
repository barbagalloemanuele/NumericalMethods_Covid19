---
name: Step 1 - Acquisizione Dati Covid
description: Istruzioni su come acquisire, filtrare (Prima Ondata) e processare in forma matriciale/spaziale i dati della Protezione Civile.
---

# Fase 1: Data Processing
Questa fase si occupa di recuperare i dati reali su cui calibrare il nostro modello PDE.

## Fonti
I dati vengono scaricati dinamicamente (o in modo statico da un CSV salvato) dal repository GitHub della Protezione Civile:
`https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv`

## Filtri e Pre-processing
- **Periodo:** Limitare i dati alla **Prima Ondata**, dal `2020-02-24` al `2020-05-31`.
- **Dati Inutili:** Rimuovere le righe con "In fase di definizione/aggiornamento" (latitudine e longitudine 0 o nulle).
- **Metriche:** Ci concentriamo sui "Totale casi".
- **Spazializzazione:** Il modello PDE richiede una griglia spaziale 2D. Lo script converte le coordinate (lat, long) delle province in indici interi (x, y) di una griglia discreta (es. 50x50). Ad ogni cella della griglia viene assegnato il numero di contagi in un dato momento.

## Librerie
- `pandas`: per la manipolazione efficiente del CSV.
- `numpy`: per la creazione della griglia spaziale 2D.
