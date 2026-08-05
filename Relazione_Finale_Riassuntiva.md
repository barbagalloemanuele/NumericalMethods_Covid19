# Relazione Finale: Simulazione Spaziale dell'Epidemia COVID-19

## Il Problema Reale Risolto
L'obiettivo di questo progetto è stato quello di superare i classici modelli epidemiologici temporali (es. curve dei contagi totali) per comprendere **come** un'epidemia si diffonde geograficamente. Conoscere la dinamica spaziale di un virus permette di prevedere quali specifiche province verranno colpite nei giorni successivi, permettendo ai governi di allocare risorse ospedaliere in modo mirato e di istituire "Zone Rosse" locali invece di lockdown nazionali.

## La Soluzione Matematica Numerica
Il problema reale è stato tradotto in un problema matematico e risolto applicando sequenzialmente i concetti fondamentali dell'Analisi Numerica:

1. **Dai Dati alla Matematica (Fase 1):** Abbiamo estratto i dati grezzi della Prima Ondata del COVID-19 dal repository della Protezione Civile. Abbiamo "spazializzato" l'Italia mappando le coordinate delle province su una griglia discreta 50x50.
2. **Discretizzazione della Fisica (Fase 2):** Abbiamo modellato la diffusione del virus usando un'Equazione alle Derivate Parziali (PDE). Per farla digerire al computer, abbiamo usato le Differenze Finite per approssimare il Laplaciano spaziale. Questo ha generato un'enorme matrice sparsa $A$ ($2500 \times 2500$), vuota al 99.8%.
3. **Risoluzione Efficiente (Fase 3):** Abbiamo evitato i disastrosi metodi di inversione diretta. Abbiamo sfruttato il fatto che il nostro Laplaciano è una matrice Simmetrica e Definita Positiva per risolvere il sistema $Ax=b$ in pochi millisecondi usando il metodo iterativo del **Gradiente Coniugato (CG)**, confrontandolo con il **GMRES**.
4. **Calibrazione con la Realtà (Fase 4):** Il nostro modello PDE aveva un'incognita: il coefficiente di diffusione ($D$) del virus. Invece di tirare a indovinare, abbiamo usato l'algoritmo di **Levenberg-Marquardt** (Ottimizzazione ai Minimi Quadrati Non Lineari) per calibrare il modello. L'algoritmo ha trovato da solo il valore di $D$ che rende la nostra simulazione il più simile possibile ai veri dati storici della Protezione Civile (Data Fitting).

## Conclusione
Questo progetto dimostra che il Calcolo Scientifico non è pura astrazione: combinando l'Algebra Lineare Sparsa (Fase 2 e 3) con l'Ottimizzazione (Fase 4), siamo riusciti a estrarre i parametri fisici nascosti di una pandemia globale semplicemente processando i dati dei contagi con i giusti metodi numerici iterativi.
