---
name: Step 2 - Discretizzazione PDE
description: Istruzioni su come costruire la matrice sparsa A (Laplaciano 2D) partendo dall'equazione di diffusione.
---

# Fase 2: Discretizzazione della PDE
In questa fase si passa dal modello matematico continuo al discreto, costruendo il sistema lineare $Ax = b$.

## Modello Fisico
L'equazione di diffusione base è:
$\frac{\partial u}{\partial t} = D \nabla^2 u$
Dove:
- $u$ è la concentrazione di contagi.
- $D$ è il coefficiente di diffusione.
- $\nabla^2$ è l'operatore Laplaciano spaziale 2D ($\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$).

## Discretizzazione (Metodo delle Differenze Finite)
Approssimiamo il Laplaciano su una griglia spaziale (quella creata in Fase 1). Usiamo la formula a 5 punti per il Laplaciano 2D:
$\nabla^2 u_{i,j} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}$

## Costruzione della Matrice Sparsa A
Per trasformare la griglia 2D in un vettore 1D (necessario per $Ax=b$), si usa il riordinamento lessicografico.
La matrice $A$ che ne deriva è:
- **Sparsa**: quasi tutti gli elementi sono 0, tranne la diagonale principale (con i -4) e alcune sottodiagonali (con gli 1).
- **Simmetrica e Definita Positiva (se negata)**.

**Attenzione:** Usa SEMPRE `scipy.sparse.diags` o `scipy.sparse.lil_matrix` per costruire $A$. Mai usare matrici dense `np.zeros()`, andrebbe fuori memoria!
