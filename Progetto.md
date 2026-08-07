# Progetto: Simulazione Spaziale di Diffusione Epidemica (Numerico vs PINN)

## 1. Descrizione dell'Idea
Il progetto si pone l'obiettivo di superare i classici modelli epidemiologici temporali (in cui si calcola solo il numero totale di infetti in un'intera nazione) per passare a un **modello spazio-temporale**. L'idea è simulare *come* il virus si sposta geograficamente su una mappa 2D col passare del tempo, ricavando i parametri di diffusione dai dati reali.

In seguito al confronto con il Professore, il progetto è stato strutturato come un'**Analisi Comparativa** tra due approcci di modellazione e risoluzione:
*   **Percorso A (Approccio Numerico Classico):** La matematica del contagio viene modellata tramite un'Equazione alle Derivate Parziali (PDE) spaziale, discretizzata su una griglia tramite differenze finite e risolta numericamente.
*   **Percorso B (Physics-Informed Neural Networks - PINN):** Il medesimo problema viene affrontato addestrando una Rete Neurale profonda la cui funzione di Loss ingloba la fisica del sistema (la PDE) e i dati storici (Data Fitting simultaneo).

## 2. Utilità e Risoluzione di un Problema Reale
Prevedere non solo *quando* ci sarà il picco di un'epidemia, ma *dove* si concentrerà, è cruciale per allocare risorse ospedaliere e istituire "Zone Rosse" mirate.
Inoltre, il progetto risponde alla moderna sfida del Calcolo Scientifico: determinare se e quando gli algoritmi di Deep Learning (PINN) possano essere più veloci, flessibili o accurati rispetto all'algebra lineare sparsa classica nel risolvere problemi differenziali fisici.

### Da dove ottenere i dati e quale periodo analizzare?
Utilizzeremo i **dati storici COVID-19 della Protezione Civile Italiana** (GitHub ufficiale).
Ci focalizzeremo sulla **Prima Ondata (Febbraio - Maggio 2020)**. Matematicamente, è il periodo perfetto: il virus è partito da focolai localizzati (es. Codogno) e si è propagato a "macchia d'olio", creando gradienti spaziali ideali per essere catturati dai solutori e calibrati tramite l'ottimizzazione ai minimi quadrati / backpropagation.

## 3. Argomenti Teorici del Corso Toccati
Il progetto copre trasversalmente quasi tutto il programma del corso, unendo i vari moduli:

### A. Metodi Numerici su Matrici Sparse (Percorso Classico)
*   **Discretizzazione PDE:** L'equazione di diffusione spaziale (Laplaciano) viene discretizzata a differenze finite generando un enorme sistema lineare sparso $Ax=b$.
*   **Solutori Iterativi:** Il sistema viene risolto implementando e confrontando **Conjugate Gradient (CG)** e **GMRES**, evitando volutamente inefficaci metodi diretti di inversione.
*   **Ottimizzazione e Minimi Quadrati:** Il parametro di diffusione incognito $D$ viene ricercato calibrando il modello sui dati reali tramite l'algoritmo di **Levenberg-Marquardt** (Data Fitting ai minimi quadrati non lineari).

### B. Machine Learning per il Calcolo Scientifico (Percorso Deep Learning)
*   **Physics-Informed Neural Networks:** Implementazione di una PINN in PyTorch.
*   **Loss Multi-Obiettivo:** La rete neurale ottimizza contemporaneamente i pesi per "matchare" i dati reali (*Data Loss*) e per rispettare l'equazione differenziale che modella il fenomeno fisico (*Physics Loss*).
*   **Scoperta di Parametri:** Il coefficiente fisico di diffusione diventa esso stesso un parametro addestrabile della rete, calcolato tramite la discesa del gradiente.

> **NOTA PER LA LETTURA:** All'interno di ogni cartella di fase (`fase_1...`, `fase_2...`, ecc.) è presente un file `Report_Fase_X.md`. Questi report fungono da capitoli della tesi e contengono una **spiegazione esplicativa dettagliata e riga-per-riga** di tutto il codice sorgente, analizzando le motivazioni matematiche dietro ogni scelta (es. perché usare la Tanh invece della ReLU, come funziona l'Autograd rispetto allo Stencil a 5 punti, ecc.).
