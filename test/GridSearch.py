#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys

sys.path.append(r"C:\\Users\\HP\\Downloads\\progetto optimization\\test")

from NeuralNetwork import NeuralNetwork
from HeavyBall import heavy_ball_optimization
from DeflectedSubgradient import deflected_subgradient_optimization


# In[4]:


import numpy as np
import itertools
import pandas as pd

def grid_search_nn(model_class, X, y, alpha_values, mu_values, gamma_values, lambda_values, method="HB", max_iterations=1000, epsilon=1e-4):
    """
    Esegue una Grid Search per ottimizzare gli iperparametri della rete neurale utilizzando
    l'algoritmo Heavy Ball o Deflected Subgradient.

    Args:
        model_class: Classe della rete neurale da ottimizzare.
        X: Input dei dati di addestramento.
        y: Output dei dati di addestramento (one-hot encoded).
        alpha_values: Lista di valori per il learning rate α.
        mu_values: Lista di valori per il momentum (solo per Heavy Ball).
        gamma_values: Lista di valori per gamma (solo per Deflected Subgradient).
        lambda_values: Lista di valori per il parametro di regolarizzazione L1.
        method: Metodo di ottimizzazione da usare ("HB" per Heavy Ball, "DSG" per Deflected Subgradient).
        max_iterations: Numero massimo di iterazioni.

    Returns:
        Una lista con le 10 migliori configurazioni ordinate per loss finale.
    """
    
    results = []

    # Generazione combinazioni parametri
    if method == "HB":
        param_grid = itertools.product(alpha_values, mu_values, lambda_values)
    elif method == "DSG":
        param_grid = itertools.product(alpha_values, gamma_values, lambda_values)
    else:
        raise ValueError("Metodo non riconosciuto. Usa 'HB' o 'DSG'.")

    total_tests = len(list(param_grid))
    test_counter = 1  # Contatore progressivo
    
    print(f"\n Inizio Grid Search con {total_tests} combinazioni...\n")
    
    for params in itertools.product(alpha_values, mu_values if method == "HB" else gamma_values, lambda_values):
        alpha, param, l1_lambda = params  # param = mu per HB, gamma per DSG

        print(f"Test {test_counter}/{total_tests} | Metodo: {method} | α={alpha} | {'μ' if method=='HB' else 'γ'}={param} | λ={l1_lambda}")

        # Inizializza la rete neurale con i parametri correnti
        model = model_class(input_size=X.shape[1], hidden_size=32, output_size=10, l1_lambda=l1_lambda)

        # Ottimizza il modello con Heavy Ball o Deflected Subgradient
        if method == "HB":
            optimized_model = heavy_ball_optimization(model, X, y, alpha, param, max_iterations=max_iterations, epsilon=epsilon)
        else:
            optimized_model = deflected_subgradient_optimization(model, X, y, alpha, param, max_iterations=max_iterations, epsilon=epsilon)

        # Recupera le metriche di valutazione dal modello ottimizzato
        loss = optimized_model.evaluate(X, y)
        convergence_speed = getattr(optimized_model, 'convergence_speed', None)
        unstability = getattr(optimized_model, 'unstability', None)
        status = getattr(optimized_model, 'solution_status', "Unknown")
        num_iterations = getattr(optimized_model, 'num_iterations', max_iterations)  # Numero iterazioni usate

        # Salva i risultati
        results.append({
            "method": "Heavy Ball" if method == "HB" else "Deflected Subgradient",
            "alpha": alpha,
            "mu/gamma": param,
            "lambda": l1_lambda,
            "loss_final": loss,
            "convergence_speed": convergence_speed,
            "unstability": unstability,
            "num_iterations": num_iterations,  # Numero di iterazioni
            "status": status
        })

        print(f" → Iterazioni effettuate: {num_iterations}\n")

        test_counter += 1

    # Converti i risultati in un DataFrame e ordina per loss finale
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="loss_final", ascending=True)  # Ordinato per loss minore

    # Seleziona le migliori 10 configurazioni
    top_10_results = df_results.head(10)

    return top_10_results


# In[ ]:





# In[ ]:




