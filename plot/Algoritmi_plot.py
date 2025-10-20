#!/usr/bin/env python
# coding: utf-8

weights_path = "C:\\Users\\HP\\Downloads\\progetto optimization\\pesi e bias generati"


#CARICAMENTO HEAVY BALL CON LOSSES COME OUTPUT
import numpy as np
import os
import time

def heavy_ball_optimization(model, X, y, alpha, mu, epsilon=1e-6, max_iterations=1000, weights_path=weights_path):
    """
    Ottimizza i parametri della rete neurale usando l'algoritmo Heavy Ball e calcola metriche di valutazione.

    Args:
        model: Istanza della rete neurale.
        X: Input del dataset.
        y: Output del dataset.
        alpha: Learning rate.
        mu: Momentum.
        epsilon: Soglia per i criteri di arresto.
        max_iterations: Numero massimo di iterazioni.
        weights_path: Percorso dei pesi salvati in .npy per inizializzare il modello.

    Returns:
        model: Modello ottimizzato.
    """
    # Carica i pesi salvati se specificato
    if weights_path and os.path.exists(weights_path):
        model.W1 = np.load(os.path.join(weights_path, "W1.npy"))
        model.b1 = np.load(os.path.join(weights_path, "b1.npy"))
        model.W2 = np.load(os.path.join(weights_path, "W2.npy"))
        model.b2 = np.load(os.path.join(weights_path, "b2.npy"))

      # Inizializzazione delle variabili
    dW1_prev, db1_prev = np.zeros_like(model.W1), np.zeros_like(model.b1)
    dW2_prev, db2_prev = np.zeros_like(model.W2), np.zeros_like(model.b2)
    losses = []  # Lista per tracciare le perdite
    unstabilities = []  # Lista per tracciare l'instabilità
    stop_count = 0  # Contatore per il criterio di stagnazione
    start_time = time.time()

    # Stampa dell'intestazione
    print(f"{'Iterazione':<12}{'Loss':<15}{'Instabilità':<15}{'Velocità Convergenza':<25}{'Norma Gradiente':<20}")
    print("-" * 90)

    for iteration in range(max_iterations):
        # Salva i valori correnti dei parametri prima dell'aggiornamento
        W1_current, b1_current = model.W1.copy(), model.b1.copy()
        W2_current, b2_current = model.W2.copy(), model.b2.copy()

        # Forward e backward propagation
        loss = model.evaluate(X, y)
        dW1, db1, dW2, db2 = model.gradient(X, y)

        # Norma del gradiente
        gradient_norm = np.linalg.norm([np.linalg.norm(dW1), np.linalg.norm(dW2), 
                                        np.linalg.norm(db1), np.linalg.norm(db2)])
        
        # Aggiungi la loss corrente alla lista delle perdite
        losses.append(loss)
        
        # Calcolo dell'instabilità
        if iteration > 1:  # Almeno due valori per calcolare l'instabilità
            unstability = (10 ** 5) * sum(
                max(0, (losses[i] - losses[i - 1]) / max(losses[i - 1], 1e-12)) for i in range(1, len(losses))
            ) / len(losses)
        else:
            unstability = 0
            
        unstabilities.append(unstability)

        # Calcolo della velocità di convergenza
        p = 1  # 1% di tolleranza
        min_loss = min(losses)
        convergence_iteration = next((i for i, l in enumerate(losses) if l <= min_loss * (1 + p / 100)), len(losses) - 1)
        convergence_speed = 100 - ((convergence_iteration * 100) / max_iterations)


        # Aggiornamento dei pesi con momentum corretto
        dW1_update = mu * dW1_prev - alpha * dW1
        db1_update = mu * db1_prev - alpha * db1
        dW2_update = mu * dW2_prev - alpha * dW2
        db2_update = mu * db2_prev - alpha * db2

        # Aggiorna i parametri
        model.W1 += dW1_update
        model.b1 += db1_update
        model.W2 += dW2_update
        model.b2 += db2_update
        
        # Aggiorna le variabili per momentum
        dW1_prev, db1_prev, dW2_prev, db2_prev = dW1_update, db1_update, dW2_update, db2_update

        # Stampa dettagli dell'iterazione
        print(f"{iteration:<12}{loss:<15.8f}{unstability:<15.8f}{convergence_speed:<25.2f}{gradient_norm:<20.8f}")

        # Criteri di arresto
        if gradient_norm < epsilon:  # Norma del gradiente
            print(f"\nConvergenza raggiunta: Norma del gradiente sotto soglia dopo {iteration + 1} iterazioni")
            solution_status = "Ottimale"
            break

        # Stagnazione: nessuna variazione significativa per 10 iterazioni consecutive
        if iteration > 10 and np.mean(np.abs(np.diff(losses[-10:]))) < epsilon:
            print(f"\nConvergenza raggiunta: Nessuna variazione significativa nella loss per 10 iterazioni consecutive.")
            solution_status = "Stopped"
            break

        # Instabilità prolungata: media mobile delle ultime 20 iterazioni troppo alta
        if iteration > 20 and np.mean(unstabilities[-20:]) > 1000:
            print("\nEccessiva instabilità nelle ultime 20 iterazioni. Interruzione per evitare problemi di divergenza.")
            solution_status = "Unstable"
            break

    else:
        # Se termina senza soddisfare alcuna condizione
        solution_status = "Stopped"

    # Calcolo del tempo di esecuzione
    end_time = time.time()
    execution_time = end_time - start_time

    # Stampa lo stato della soluzione
    print(f"\nStato della soluzione: {solution_status}")
    print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
    print(f"Numero di iterazioni: {iteration + 1}")
    print(f"Loss finale: {loss:.8f}")
    print(f"Instabilità: {unstability: .8f}")
    print(f"Velocità di convergenza: {convergence_speed: .8f}")

    return losses, model


# In[ ]:


import numpy as np
import os
import time

def deflected_subgradient_optimization(model, X, y, alpha_init, gamma=0.5, epsilon=1e-6, max_iterations=1000, weights_path=weights_path, rho=1):
    """
    Ottimizza i parametri della rete neurale usando il metodo del Sottogradiente Deflesso
    (Deflected Subgradient Method) con step size adattivo DSS e calcola metriche di valutazione.

    Args:
        model: Istanza della rete neurale.
        X: Input del dataset.
        y: Output del dataset.
        alpha_init: Learning rate iniziale (costante per DSS).
        gamma: Parametro di deflessione (0 <= gamma <= 1).
        epsilon: Soglia per il criterio di arresto.
        max_iterations: Numero massimo di iterazioni.
        weights_path: Percorso dei pesi salvati in .npy per inizializzare il modello.
        rho: Percentuale per definire la velocità di convergenza.

    Returns:
        model: Modello ottimizzato.
    """

    # Carica i pesi salvati se specificato
    if weights_path and os.path.exists(weights_path):
        model.W1 = np.load(os.path.join(weights_path, "W1.npy"))
        model.b1 = np.load(os.path.join(weights_path, "b1.npy"))
        model.W2 = np.load(os.path.join(weights_path, "W2.npy"))
        model.b2 = np.load(os.path.join(weights_path, "b2.npy"))

    # Inizializzazione delle variabili
    dW1_prev, db1_prev = np.zeros_like(model.W1), np.zeros_like(model.b1)
    dW2_prev, db2_prev = np.zeros_like(model.W2), np.zeros_like(model.b2)
    losses = []  # Lista per tracciare le perdite
    unstabilities = []  # Lista per instabilità
    stop_count = 0  # Contatore per stagnazione
    instability_stop_count = 0  # Contatore per instabilità
    start_time = time.time()
    
    # Inizializzazione della costante per DSS
    c = alpha_init 

    # Stampa dell'intestazione
    print(f"{'Iterazione':<12}{'Loss':<15}{'Norma Subgradiente':<20}{'Instabilità':<15}{'Velocità Convergenza':<25}")
    print("-" * 90)

    # Valore ottimale della funzione (inizialmente infinito)
    f_best = float('inf')

    for iteration in range(1, max_iterations + 1):  # Iterazioni partono da 1 per DSS
        
        # Step size adattivo DSS
        alpha = c / iteration  # Diminishing step size α_i = c / i

        # Calcola la funzione di perdita corrente
        loss = model.evaluate(X, y)
        
        # Calcola i subgradienti
        gW1, gb1, gW2, gb2 = model.gradient(X, y)

        # Norma del subgradiente
        subgradient_norm = np.linalg.norm([np.linalg.norm(gW1), np.linalg.norm(gW2), 
                                           np.linalg.norm(gb1), np.linalg.norm(gb2)])
        
        # Aggiungi la loss corrente alla lista delle perdite
        losses.append(loss)

        # Calcolo dell'instabilità con la formula del professore
        if iteration > 1:
            unstability = (10 ** 5) * sum(
                max(0, (losses[i] - losses[i - 1]) / max(losses[i - 1], 1e-12)) for i in range(1, len(losses))
            ) / len(losses)
        else:
            unstability = 0
        unstabilities.append(unstability)

        # Calcolo della velocità di convergenza con la formula corretta
        p = 1  # Tolleranza percentuale
        min_loss = min(losses)
        convergence_iteration = next((i for i, val in enumerate(losses) if val <= min_loss * (1 + p / 100)), len(losses) - 1)
        convergence_speed = 100 - (convergence_iteration * 100 / max_iterations)

        # Aggiornamento della direzione deflessa
        dW1 = gamma * gW1 + (1 - gamma) * dW1_prev
        db1 = gamma * gb1 + (1 - gamma) * db1_prev
        dW2 = gamma * gW2 + (1 - gamma) * dW2_prev
        db2 = gamma * gb2 + (1 - gamma) * db2_prev

        # Aggiornamento dei parametri
        model.W1 -= alpha * dW1
        model.b1 -= alpha * db1
        model.W2 -= alpha * dW2
        model.b2 -= alpha * db2

        # Aggiorna le variabili per il passo successivo
        dW1_prev, db1_prev, dW2_prev, db2_prev = dW1, db1, dW2, db2

        # Stampa dettagli dell'iterazione
        print(f"{iteration:<12}{loss:<15.8f}{subgradient_norm:<20.8f}{unstability:<15.8f}{convergence_speed:<25.2f}")

        # Aggiorna il miglior valore della loss
        if loss < f_best:
            f_best = loss

        # Criteri di arresto
        if subgradient_norm < epsilon:
            print(f"\nConvergenza raggiunta: Norma del subgradiente sotto soglia dopo {iteration} iterazioni")
            solution_status = "Ottimale"
            break

        # Stagnazione: nessuna variazione significativa per 10 iterazioni consecutive
        if iteration > 10 and np.mean(np.abs(np.diff(losses[-10:]))) < epsilon:
            print(f"\nConvergenza raggiunta: Nessuna variazione significativa nella loss per 10 iterazioni consecutive.")
            solution_status = "Stopped"
            break

        # Instabilità prolungata: media mobile delle ultime 20 iterazioni troppo alta
        if iteration > 20 and np.mean(unstabilities[-20:]) > 1000:
            print("\nEccessiva instabilità nelle ultime 20 iterazioni. Interruzione per evitare problemi di divergenza.")
            solution_status = "Unstable"
            break

    else:
        print("Numero massimo di iterazioni raggiunto")
        solution_status = "Stopped"

    # Calcolo del tempo di esecuzione
    end_time = time.time()
    execution_time = end_time - start_time

    # Stampa lo stato della soluzione
    print(f"\nStato della soluzione: {solution_status}")
    print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
    print(f"Numero di iterazioni: {iteration}")
    print(f"Loss finale: {loss:.8f}")
    print(f"Instabilità: {unstability: .8f}")
    print(f"Velocità di convergenza: {convergence_speed: .8f}")

    return losses, model

