# Report Fase 5: Physics-Informed Neural Networks (Percorso B)

## Cosa bisogna fare in questa fase
Entriamo nel **Percorso B (Machine Learning per il Calcolo Scientifico)**. L'obiettivo è risolvere lo stesso problema della Fase 3 e 4, ma senza costruire esplicitamente l'enorme matrice sparsa $A$. 
Invece di discretizzare la fisica su una griglia fissa (Differenze Finite), utilizziamo una **Rete Neurale Artificiale** per imparare la soluzione. 
Il concetto alla base delle *Physics-Informed Neural Networks* (PINN) è brillante: istruiamo la rete neurale a minimizzare un errore (Loss) che non tiene conto solo dei dati misurati, ma che la "punisce" anche se produce output che violano le leggi della fisica (nel nostro caso, l'equazione differenziale della diffusione).

## Cosa fa il codice (`pinn_solver.py`)
Lo script utilizza `PyTorch` per definire e addestrare una Multi-Layer Perceptron (MLP) profonda.
- **La Rete:** Prende in input le coordinate spazio-temporali $(x, y, t)$ e restituisce il numero di contagi in quel punto e in quell'istante $u(x, y, t)$. La funzione di attivazione usata è la Tangente Iperbolica (`Tanh`), matematicamente essenziale perché dotata di derivate seconde continue non nulle (fondamentale per calcolare il Laplaciano nella Loss).
- **La Loss Multi-Obiettivo:** La rete è addestrata minimizzando una somma di due errori:
  1. `Data Loss`: Costringe la rete a riprodurre i dati reali storici (i contagi estratti dal DB).
  2. `Physics Loss`: Usando la derivazione automatica differenziale (`torch.autograd`), lo script calcola le derivate parziali dell'output rispetto agli input (es. derivata temporale e laplaciano spaziale), ricostruisce l'equazione di diffusione e "punisce" la rete se il risultato non fa zero.
- **Scoperta del Parametro:** L'aspetto più affascinante è il coefficiente di diffusione ($D$). Invece di essere un numero fisso, è un peso addestrabile (`nn.Parameter`). Mentre la rete cerca di minimizzare le Loss per far tornare i conti tra i dati e l'equazione, la Discesa del Gradiente (Backpropagation) corregge il valore di $D$, convergendo iterativamente verso il valore fisico reale.

## Commento all'Output
All'avvio, il parametro $D$ viene inizializzato a $1.0$ (un'ipotesi cieca). 
Con lo scorrere delle epoche di addestramento:
- La **Loss Totale** diminuisce progressivamente (segno che la rete sta trovando un compromesso tra i dati e la fisica).
- Il valore di **$D$ scoperto dalla PINN** cambia gradualmente fino a stabilizzarsi attorno al valore corretto (nella nostra simulazione di test, era nascosto a `0.35`).
- Al termine dell'esecuzione, i pesi neurali addestrati vengono salvati nel file `pinn_model.pth`, pronti per essere importati nel modulo di confronto (Fase 6) o spinti sul Cluster UNICT per l'esecuzione su larga scala.
