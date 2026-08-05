# Progetto: Simulazione Spaziale di Diffusione Epidemica tramite Metodi Numerici

## 1. Descrizione dell'Idea
Il progetto si pone l'obiettivo di superare i classici modelli epidemiologici temporali (in cui si calcola solo il numero totale di infetti in un'intera nazione) per passare a un **modello spazio-temporale**. 
L'idea è simulare *come* il virus si sposta geograficamente su una mappa 2D col passare del tempo. Questo viene matematicamente modellato tramite un'Equazione alle Derivate Parziali (PDE) di tipo diffusione-reazione. 

Per risolvere questa equazione al computer, lo spazio continuo (la mappa geografica) viene trasformato in una griglia discreta (Discretizzazione tramite Differenze Finite). Questa operazione matematica converte la complessa equazione differenziale in un **sistema lineare di enormi dimensioni ($Ax = b$)**, la cui matrice $A$ risulta essere molto grande ma "sparsa" (piena di zeri). 
Per risolvere questo sistema verranno utilizzati e confrontati vari **Metodi Iterativi**. Infine, il modello verrà calibrato sui dati reali tramite i **Minimi Quadrati**.

## 2. Utilità e Risoluzione di un Problema Reale
Prevedere non solo *quando* ci sarà il picco di un'epidemia, ma *dove* si concentrerà, è un problema reale di fondamentale importanza per:
*   **Allocazione delle risorse:** Sapere in anticipo quali province avranno bisogno di più posti letto in terapia intensiva.
*   **Restrizioni mirate:** Permette ai decisori politici di istituire "Zone Rosse" mirate invece di lockdown nazionali indiscriminati, salvaguardando l'economia delle regioni non a rischio.

### Da dove ottenere i dati e quale periodo analizzare?
Utilizzeremo i **dati storici COVID-19 della Protezione Civile Italiana**, reperibili pubblicamente dal loro repository GitHub ufficiale. 
*   **Perché:** Offrono la **granularità spaziale** necessaria per questo progetto (dati dei contagi suddivisi per singola *Provincia* e *Regione* ogni giorno). 
*   **Periodo Scelto:** Il progetto si focalizzerà sulla **Prima Ondata (Febbraio - Maggio 2020)**. Dal punto di vista matematico della diffusione (PDE), è il periodo perfetto: il virus è partito da focolai molto localizzati (es. Codogno in Lombardia, Vo' in Veneto) e si è propagato spazialmente e gradualmente verso il resto d'Italia a "macchia d'olio". Questo crea un chiaro gradiente spaziale, ideale per essere catturato dal modello di diffusione e calibrato dai minimi quadrati.

## 3. Argomenti Teorici del Corso Toccati

Il progetto copre trasversalmente quasi tutto il programma del Prof. Boscarino, unendo i vari moduli in un'unica pipeline logica:

### A. Discretizzazione di Equazioni alle Derivate Parziali (PDE)
*   **Concetto:** La PDE della diffusione termica/epidemica viene discretizzata usando il metodo delle differenze finite su una griglia 2D.
*   **Risultato:** Trasformazione di un problema fisico in un problema di Algebra Lineare (costruzione della matrice sparsa $A$). È lo stesso approccio iniziale descritto dal professore nel progetto sul *Quantum Computing*.

### B. Metodi Iterativi per Sistemi Lineari
*   **Concetto:** Poiché la mappa geografica genererà migliaia di nodi, il sistema lineare $Ax = b$ risultante sarà troppo grande e costoso da risolvere con metodi diretti (es. Eliminazione di Gauss).
*   **Applicazione:** Implementazione e confronto dei metodi iterativi visti a lezione, come il **Metodo del Gradiente Coniugato (CG)** o il **GMRES** (Generalized Minimal Residual method), analizzandone velocità di convergenza e residuo.

### C. Minimi Quadrati e Ottimizzazione
*   **Concetto:** Il modello PDE possiede dei parametri fisici incogniti, primo fra tutti il *Coefficiente di Diffusione* (che indica quanto velocemente le persone si spostano e portano il virus nei territori limitrofi).
*   **Applicazione:** Avendo a disposizione i dati reali (es. casi in Lombardia vs casi in Veneto nel tempo), si definirà una funzione di costo (l'errore tra i dati predetti dalla PDE e i dati veri della Protezione Civile). Utilizzando i **Minimi Quadrati**, cercheremo numericamente il valore ottimale di questo coefficiente che minimizza l'errore, effettuando un vero e proprio *Data Fitting*.
