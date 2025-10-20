#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time

def heavy_ball_method(model, alpha, mu, epsilon=1e-12, max_iterations=1000):
    """
    Ottimizza una funzione generica usando l'algoritmo Heavy Ball e calcola metriche di valutazione.

    Args:
        model: Istanza della funzione da ottimizzare, con metodi evaluate() e gradient().
        alpha: Learning rate.
        mu: Momentum.
        epsilon: Soglia per i criteri di arresto.
        max_iterations: Numero massimo di iterazioni.

    Returns:
        model: Modello ottimizzato.
    """
    # Inizializzazione delle variabili
    dW_prev = np.zeros_like(model.W1)  # Variabile per il momentum
    losses = []  # Lista per tracciare la perdita
    unstabilities = []  # Lista per tracciare l'instabilità
    start_time = time.time()
    stop_count = 0  # Contatore per stagnazione
    instability_stop_count = 0  # Contatore per instabilità

    # Stampa dell'intestazione
    print(f"{'Iterazione':<12}{'Loss':<15}{'Instabilità':<15}{'Velocità Convergenza':<25}{'Norma Gradiente':<20}")
    print("-" * 90)

    for iteration in range(max_iterations):
        # Calcola la perdita corrente e il gradiente
        loss = model.evaluate()
        gradient = np.array(model.gradient())  # Gradiente come array

        # Controllo dimensioni gradienti
        if gradient.shape != model.W1.shape:
            raise ValueError(f"Dimensioni del gradiente ({gradient.shape}) non compatibili con W1 ({model.W1.shape}).")

        # Norma del gradiente
        gradient_norm = np.linalg.norm(gradient)

        # Aggiungi la loss corrente alla lista delle perdite
        losses.append(loss)

        # Calcolo dell'instabilità 
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

        # Aggiornamento dei pesi con momentum (ordine corretto per Heavy Ball)
        dW_update = mu * dW_prev - alpha * gradient  

        # Aggiorna i parametri
        model.W1 += dW_update

        # Aggiorna la variabile di momentum
        dW_prev = dW_update

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
        print("Numero massimo di iterazioni raggiunto")
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
    
        # Aggiunge le metriche al modello
    model.convergence_speed = convergence_speed
    model.unstability = unstability
    model.solution_status = solution_status

    return model


# In[2]:


import numpy as np
import time

def deflected_subgradient_method(model, alpha, gamma=0.5, epsilon=1e-12, max_iterations=1000, rho=1):
    """
    Ottimizza i parametri di una funzione usando il metodo del Sottogradiente Deflesso 
    (Deflected Subgradient Method) con step size adattivo DSS e metriche di valutazione.

    Args:
        model: Oggetto che rappresenta la funzione da ottimizzare, con metodi `evaluate` e `gradient`.
        alpha_init: Learning rate iniziale.
        gamma: Parametro di deflessione (0 <= gamma <= 1).
        epsilon: Soglia per il criterio di arresto.
        max_iterations: Numero massimo di iterazioni.
        rho: Percentuale per definire la velocità di convergenza.

    Returns:
        model: Modello ottimizzato.
    """
    
    # Inizializzazione delle variabili
    dW_prev = np.zeros_like(model.W1)  # Variabile per la direzione deflessa
    losses = []  # Lista per tracciare le perdite
    unstabilities = []  # Lista per tracciare l'instabilità
    start_time = time.time()
    stop_count = 0  # Contatore per stagnazione
    instability_stop_count = 0  # Contatore per instabilità
    
    # Inizializzazione della costante per DSS
    c = alpha 

    # Stampa dell'intestazione
    print(f"{'Iterazione':<12}{'Loss':<15}{'Norma Subgradiente':<20}{'Instabilità':<15}{'Velocità Convergenza':<25}")
    print("-" * 90)

    for iteration in range(1, max_iterations + 1):  # Iterazioni partono da 1 per DSS
        
        # Step size adattivo DSS
        alpha = c / iteration  

        # Calcola la funzione di perdita corrente
        loss = model.evaluate()
        
        # Calcola il subgradiente
        gradient = np.array(model.gradient())

        # Controllo dimensioni gradienti
        if gradient.shape != model.W1.shape:
            raise ValueError(f"Dimensioni del gradiente ({gradient.shape}) non compatibili con W1 ({model.W1.shape}).")

        # Norma del subgradiente
        subgradient_norm = np.linalg.norm(gradient)

        # Aggiungi la loss corrente alla lista delle perdite
        losses.append(loss)

        # Calcolo dell'instabilità
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
        dW_update = gamma * gradient + (1 - gamma) * dW_prev

        # Aggiorna i parametri
        model.W1 -= alpha * dW_update

        # Aggiorna la variabile per il passo successivo
        dW_prev = dW_update

        # Stampa dettagli dell'iterazione
        print(f"{iteration:<12}{loss:<15.8f}{subgradient_norm:<20.8f}{unstability:<15.8f}{convergence_speed:<25.2f}")

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

    # Calcolo del tempo di esecuzione totale
    end_time = time.time()
    execution_time = end_time - start_time

    # Stampa lo stato della soluzione
    print(f"\nStato della soluzione: {solution_status}")
    print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
    print(f"Numero di iterazioni: {iteration}")
    print(f"Loss finale: {loss:.8f}")
    print(f"Instabilità: {unstability: .8f}")
    print(f"Velocità di convergenza: {convergence_speed: .8f}")
    
        # Aggiunge le metriche al modello
    model.convergence_speed = convergence_speed
    model.unstability = unstability
    model.solution_status = solution_status

    return model


# In[3]:


# Definizione del modello per la funzione di Matyas
class MatyasFunction:
    def __init__(self):
        self.W1 = np.array([4.0, 4.0])  # Inizializzazione dei parametri x e y (punto iniziale)

    def evaluate(self):
        """Calcola il valore della funzione di Matyas dato un punto (x, y)."""
        x, y = self.W1
        return 0.26 * (x**2 + y**2) - 0.48 * x * y

    def gradient(self):
        """Calcola il gradiente della funzione di Matyas rispetto a (x, y)."""
        x, y = self.W1
        dx = 0.52 * x - 0.48 * y
        dy = 0.52 * y - 0.48 * x
        return np.array([dx, dy])  # Ritorna il gradiente come vettore


# In[4]:


# Definizione del modello per la funzione di Himmbleau
class HimmbleauFunction:
    def __init__(self):
        self.W1 = np.array([4.0, 4.0])  # Inizializzazione dei parametri x e y (punto iniziale)

    def evaluate(self):
        """
        Calcola il valore della funzione di Himmelblau dato un punto (x, y).
        La funzione di Himmelblau è definita come:
        f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
        """
        x, y = self.W1
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    def gradient(self):
        """
        Calcola il gradiente della funzione di Himmelblau rispetto a (x, y).
        Le derivate parziali sono:
        df/dx = 4 * (x^2 + y - 11) * x + 2 * (x + y^2 - 7)
        df/dy = 2 * (x^2 + y - 11) + 4 * (x + y^2 - 7) * y
        """
        x, y = self.W1
        dx = 4 * (x**2 + y - 11) * x + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 4 * (x + y**2 - 7) * y
        return np.array([dx, dy])  # Ritorna il gradiente come vettore


# In[ ]:





# In[5]:


import pandas as pd
import numpy as np
import time

def grid_search(function, method, start_point, alpha_values, momentum_values=None, gamma_values=None):
    """
    Esegue una grid search per trovare i migliori parametri (alpha, mu o gamma)
    per ottimizzare una funzione utilizzando un metodo di ottimizzazione dato.

    Args:
        function: Classe della funzione obiettivo (MatyasFunction o HimmelblauFunction).
        method: Metodo di ottimizzazione (heavy_ball_method o deflected_subgradient_method).
        start_point: Punto iniziale per la funzione (array numpy).
        alpha_values: Lista di valori di learning rate (alpha).
        momentum_values: Lista di valori di momentum (mu) (per Heavy Ball).
        gamma_values: Lista di valori di deflessione (gamma) (per Deflected Subgradient).

    Returns:
        DataFrame Pandas con i risultati della Grid Search.
    """
    results = []

    # Itera attraverso tutte le combinazioni di parametri
    for alpha in alpha_values:
        if momentum_values:  # Per Heavy Ball
            for mu in momentum_values:
                print(f"\n🔹 Eseguendo: Metodo=Heavy Ball, Funzione={function.__name__}, Alpha={alpha}, Mu={mu}")
                
                # Reset della funzione e impostazione del punto iniziale
                model = function()  
                model.W1 = np.copy(start_point)

                # Esegui l'ottimizzazione
                start_time = time.time()
                optimized_model = method(model, alpha, mu)
                execution_time = time.time() - start_time

                # Recupera le metriche di valutazione
                loss = optimized_model.evaluate()
                convergence_speed = getattr(optimized_model, 'convergence_speed', None)
                unstability = getattr(optimized_model, 'unstability', None)
                status = getattr(optimized_model, 'solution_status', "Unknown")

                # Salva i risultati
                results.append({
                    "method": "Heavy Ball",
                    "alpha": alpha,
                    "mu": mu,
                    "gamma": None,
                    "loss_finale": loss,
                    "convergence_speed": convergence_speed,
                    "unstability": unstability,
                    "execution_time": execution_time,
                    "status": status
                })
        
        elif gamma_values:  # Per Deflected Subgradient
            for gamma in gamma_values:
                print(f"\n🔹 Eseguendo: Metodo=Deflected Subgradient, Funzione={function.__name__}, Alpha={alpha}, Gamma={gamma}")
                
                # Reset della funzione e impostazione del punto iniziale
                model = function()  
                model.W1 = np.copy(start_point)

                # Esegui l'ottimizzazione
                start_time = time.time()
                optimized_model = method(model, alpha, gamma)
                execution_time = time.time() - start_time

                # Recupera le metriche di valutazione
                loss = optimized_model.evaluate()
                convergence_speed = getattr(optimized_model, 'convergence_speed', None)
                unstability = getattr(optimized_model, 'unstability', None)
                status = getattr(optimized_model, 'solution_status', "Unknown")

                # Salva i risultati
                results.append({
                    "method": "Deflected Subgradient",
                    "alpha": alpha,
                    "mu": None,
                    "gamma": gamma,
                    "loss_finale": loss,
                    "convergence_speed": convergence_speed,
                    "unstability": unstability,
                    "execution_time": execution_time,
                    "status": status
                })

    # Converti i risultati in un DataFrame Pandas
    df = pd.DataFrame(results)

    # Filtra solo le configurazioni valide (esclude quelle instabili)
    df_valid = df[df["status"] != "Unstable"]

    # Ordina per Loss Finale (crescente), poi per Convergence Speed (decrescente), e Execution Time (crescente)
    df_valid = df_valid.sort_values(by=["loss_finale", "convergence_speed", "execution_time"], ascending=[True, False, True])

    # Stampa la tabella con i risultati migliori
    print("\n Risultati Grid Search:")
    print(df_valid.head(10))  # Mostra le migliori 10 configurazioni

    # Mostra i migliori parametri trovati
    best_config = df_valid.iloc[0] if not df_valid.empty else None

    if best_config is not None:
        print("\n Miglior configurazione trovata:")
        print(best_config)

    return df_valid


# In[6]:


# Parametri per Grid Search
start_point = np.array([4.0, 4.0])  
alpha_values = [0.01, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0]
momentum_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  


# In[7]:


print("\n Grid Search - Heavy Ball Method (Matyas)")
df_hb_matyas = grid_search(MatyasFunction, heavy_ball_method, start_point, alpha_values, momentum_values=momentum_values)


# In[8]:


print("\n Grid Search - Heavy Ball Method (Himmelblau)")
df_hb_himmelblau = grid_search(HimmbleauFunction, heavy_ball_method, start_point, alpha_values, momentum_values=momentum_values)


# In[9]:


print("\n Grid Search - Deflected Subgradient Method (Matyas)")
df_ds_matyas = grid_search(MatyasFunction, deflected_subgradient_method, start_point, alpha_values, gamma_values=gamma_values)


# In[10]:


print("\n Grid Search - Deflected Subgradient Method (Himmelblau)")
df_ds_himmelblau = grid_search(HimmbleauFunction, deflected_subgradient_method, start_point, alpha_values, gamma_values=gamma_values)


# In[11]:


#con alpha costante
import numpy as np
import time

def deflected_subgradient_method(model, alpha, gamma=0.5, epsilon=1e-12, max_iterations=1000, rho=1):
    """
    Ottimizza i parametri di una funzione usando il metodo del Sottogradiente Deflesso 
    (Deflected Subgradient Method) con step size adattivo DSS e metriche di valutazione.

    Args:
        model: Oggetto che rappresenta la funzione da ottimizzare, con metodi `evaluate` e `gradient`.
        alpha_init: Learning rate iniziale.
        gamma: Parametro di deflessione (0 <= gamma <= 1).
        epsilon: Soglia per il criterio di arresto.
        max_iterations: Numero massimo di iterazioni.
        rho: Percentuale per definire la velocità di convergenza.

    Returns:
        model: Modello ottimizzato.
    """
    
    # Inizializzazione delle variabili
    dW_prev = np.zeros_like(model.W1)  # Variabile per la direzione deflessa
    losses = []  # Lista per tracciare le perdite
    unstabilities = []  # Lista per tracciare l'instabilità
    start_time = time.time()
    stop_count = 0  # Contatore per stagnazione
    instability_stop_count = 0  # Contatore per instabilità
    

    # Stampa dell'intestazione
    print(f"{'Iterazione':<12}{'Loss':<15}{'Norma Subgradiente':<20}{'Instabilità':<15}{'Velocità Convergenza':<25}")
    print("-" * 90)

    for iteration in range(1, max_iterations + 1):  
        

        # Calcola la funzione di perdita corrente
        loss = model.evaluate()
        
        # Calcola il subgradiente
        gradient = np.array(model.gradient())

        # Controllo dimensioni gradienti
        if gradient.shape != model.W1.shape:
            raise ValueError(f"Dimensioni del gradiente ({gradient.shape}) non compatibili con W1 ({model.W1.shape}).")

        # Norma del subgradiente
        subgradient_norm = np.linalg.norm(gradient)

        # Aggiungi la loss corrente alla lista delle perdite
        losses.append(loss)

        # Calcolo dell'instabilità 
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
        dW_update = gamma * gradient + (1 - gamma) * dW_prev

        # Aggiorna i parametri
        model.W1 -= alpha * dW_update

        # Aggiorna la variabile per il passo successivo
        dW_prev = dW_update

        # Stampa dettagli dell'iterazione
        print(f"{iteration:<12}{loss:<15.8f}{subgradient_norm:<20.8f}{unstability:<15.8f}{convergence_speed:<25.2f}")

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

    # Calcolo del tempo di esecuzione totale
    end_time = time.time()
    execution_time = end_time - start_time

    # Stampa lo stato della soluzione
    print(f"\nStato della soluzione: {solution_status}")
    print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
    print(f"Numero di iterazioni: {iteration}")
    print(f"Loss finale: {loss:.8f}")
    print(f"Instabilità: {unstability: .8f}")
    print(f"Velocità di convergenza: {convergence_speed: .8f}")
    
        # Aggiunge le metriche al modello
    model.convergence_speed = convergence_speed
    model.unstability = unstability
    model.solution_status = solution_status

    return model


# In[12]:


print("\n Grid Search - Deflected Subgradient Method (Matyas)")
df_ds_matyas = grid_search(MatyasFunction, deflected_subgradient_method, start_point, alpha_values, gamma_values=gamma_values)


# In[13]:


print("\n Grid Search - Deflected Subgradient Method (Himmelblau)")
df_ds_himmelblau = grid_search(HimmbleauFunction, deflected_subgradient_method, start_point, alpha_values, gamma_values=gamma_values)


# In[ ]:





# In[30]:


import numpy as np
import time

def heavy_ball_method(model, alpha, mu, epsilon=1e-12, max_iterations=1000):
    """
    Ottimizza i parametri della funzione usando l'algoritmo Heavy Ball e calcola metriche di valutazione.
    """
    # Inizializzazione delle variabili
    dW_prev = np.zeros_like(model.W1)  
    losses = []  
    descendent_path = []  
    start_time = time.time()
    stop_count = 0  

    # Stampa dell'intestazione
    print(f"{'Iterazione':<12}{'Loss':<15}{'Instabilità':<15}{'Velocità Convergenza':<25}{'Norma Gradiente':<20}")
    print("-" * 90)

    for iteration in range(1, max_iterations + 1):
        # Calcola la perdita e il gradiente
        loss = model.evaluate()
        gradient = np.array(model.gradient())  

        # Assicurati che gradient abbia la stessa forma di W1
        if gradient.shape != model.W1.shape:
            raise ValueError(f"Dimensioni del gradiente ({gradient.shape}) non compatibili con W1 ({model.W1.shape}).")

        # Norma del gradiente
        gradient_norm = np.linalg.norm(gradient)

        # Aggiungi la loss corrente alla lista delle perdite
        losses.append(loss)

        # Traccia il percorso di discesa
        descendent_path.append(np.copy(model.W1))

        # Calcolo dell'instabilità
        if iteration > 1:  
            instability = (10 ** 5) * sum(
                max(0, (losses[i] - losses[i - 1]) / max(losses[i - 1], 1e-12)) for i in range(1, len(losses))
            ) / len(losses)
        else:
            instability = 0

        # Calcolo della velocità di convergenza
        p = 1  
        threshold = min(losses) * (1 + p / 100)
        try:
            index = next(i for i, val in enumerate(losses) if val <= threshold)
            convergence_speed = 100 - (index * 100 / max_iterations)
        except StopIteration:
            convergence_speed = 0

        # Aggiornamento dei pesi con momentum
        dW_update = mu * dW_prev - alpha * gradient  

        # Aggiorna i parametri
        model.W1 += dW_update

        # Aggiorna la variabile di momentum
        dW_prev = dW_update

        # Stampa dettagli dell'iterazione
        print(f"{iteration:<12}{loss:<15.8f}{instability:<15.8f}{convergence_speed:<25.2f}{gradient_norm:<20.8f}")

        # Criteri di arresto
        if gradient_norm < epsilon:  
            print(f"\nConvergenza raggiunta: Norma del gradiente sotto soglia dopo {iteration} iterazioni")
            solution_status = "Ottimale"
            break

        if iteration > 10 and np.mean(np.abs(np.diff(losses[-10:]))) < epsilon:
            print(f"\nConvergenza raggiunta: Nessuna variazione significativa nella loss per 10 iterazioni consecutive.")
            solution_status = "Stopped"
            break

        if iteration > 20 and np.mean([instability]) > 1000:
            print("\nEccessiva instabilità nelle ultime 20 iterazioni. Interruzione per evitare problemi di divergenza.")
            solution_status = "Unstable"
            break

    else:
        print("Numero massimo di iterazioni raggiunto")
        solution_status = "Stopped"

    # Tempo di esecuzione
    end_time = time.time()
    execution_time = end_time - start_time

    # Stampa riepilogo finale
    print(f"\nStato della soluzione: {solution_status}")
    print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
    print(f"Numero di iterazioni: {iteration}")
    print(f"Loss finale: {loss:.8f}")
    print(f"Instabilità: {instability:.8f}")
    print(f"Velocità di convergenza: {convergence_speed:.8f}")

    return losses, descendent_path

def deflected_subgradient_method(model, alpha, gamma=0.5, epsilon=1e-12, max_iterations=1000):
    """
    Ottimizza i parametri di una funzione usando il metodo del Sottogradiente Deflesso 
    con step-size decrescente (DSS) e metriche di valutazione simili all'algoritmo Heavy Ball.
    
    Args:
        model: Oggetto che rappresenta la funzione da ottimizzare, con metodi `evaluate` e `gradient`.
        alpha_init: Learning rate iniziale.
        gamma: Parametro di deflessione (0 <= gamma <= 1).
        epsilon: Soglia per il criterio di arresto.
        max_iterations: Numero massimo di iterazioni.

    Returns:
        losses: Lista delle perdite a ogni iterazione.
        descendent_path: Lista degli stati di W1 a ogni iterazione.
    """
    # Inizializzazione delle variabili
    dW_prev = np.zeros_like(model.W1)  
    losses = []  
    descendent_path = []  
    start_time = time.time()
    stop_count = 0  

    # Stampa dell'intestazione
    print(f"{'Iterazione':<12}{'Loss':<15}{'Instabilità':<15}{'Velocità Convergenza':<25}{'Norma SubGradiente':<20}")
    print("-" * 70)

    for iteration in range(1, max_iterations + 1):
        # Calcola il valore della funzione obiettivo
        loss = model.evaluate()
        gradient = np.array(model.gradient())

        # Assicurati che gradient abbia la stessa forma di W1
        if gradient.shape != model.W1.shape:
            raise ValueError(f"Dimensioni del gradiente ({gradient.shape}) non compatibili con W1 ({model.W1.shape}).")

        # Norma del subgradiente
        subgradient_norm = np.linalg.norm(gradient)

        # Aggiungi la loss corrente alla lista delle perdite
        losses.append(loss)

        # Traccia il percorso di discesa
        descendent_path.append(np.copy(model.W1))

        # Calcolo dell'instabilità
        if iteration > 1:
            instability = (10 ** 5) * sum(
                max(0, (losses[i] - losses[i - 1]) / max(losses[i - 1], 1e-12)) for i in range(1, len(losses))
            ) / len(losses)
        else:
            instability = 0
            
            
        # Calcolo della velocità di convergenza
        p = 1  
        threshold = min(losses) * (1 + p / 100)
        try:
            index = next(i for i, val in enumerate(losses) if val <= threshold)
            convergence_speed = 100 - (index * 100 / max_iterations)
        except StopIteration:
            convergence_speed = 0

        # Aggiornamento della direzione deflessa
        dW_update = gamma * gradient + (1 - gamma) * dW_prev  

        # Aggiornamento dei pesi 
        model.W1 -= alpha * dW_update  

        # Aggiorna la variabile per il passo successivo
        dW_prev = dW_update

        # Stampa dettagli dell'iterazione
        print(f"{iteration:<12}{loss:<15.8f}{instability:<15.8f}{convergence_speed:<25.2f}{subgradient_norm:<20.8f}")

        # Criteri di arresto
        if subgradient_norm < epsilon:
            print(f"\n Convergenza raggiunta: Norma del subgradiente sotto soglia dopo {iteration} iterazioni")
            solution_status = "Ottimale"
            break

        if iteration > 10 and np.mean(np.abs(np.diff(losses[-10:]))) < epsilon:
            print(f"\n Convergenza raggiunta: Nessuna variazione significativa nella loss per 10 iterazioni consecutive.")
            solution_status = "Stopped"
            break

        if iteration > 20 and np.mean([instability]) > 1000:
            print("\n Eccessiva instabilità nelle ultime 20 iterazioni. Interruzione per evitare problemi di divergenza.")
            solution_status = "Unstable"
            break

    else:
        print(" Numero massimo di iterazioni raggiunto")
        solution_status = "Stopped"

    # Tempo di esecuzione
    end_time = time.time()
    execution_time = end_time - start_time

    # Stampa riepilogo finale
    print(f"\n Stato della soluzione: {solution_status}")
    print(f" Tempo di esecuzione: {execution_time:.2f} secondi")
    print(f" Numero di iterazioni: {iteration}")
    print(f" Loss finale: {loss:.8f}")
    print(f" Instabilità: {instability:.8f}")

    return losses, descendent_path


# In[27]:


mat=MatyasFunction()
him=HimmbleauFunction()

losses_hb_matyas,_= heavy_ball_method(mat, 1, 0.6)
losses_hb_himmelblau,_ = heavy_ball_method(him, 0.01, 0.1)
losses_dsg_matyas,_ = deflected_subgradient_method(mat, 1, gamma=0.2)
losses_dsg_himmelbleau,_ = deflected_subgradient_method(him, 0.02, gamma=0.9)


# In[29]:


import pandas as pd

data = {
    "Metodo": ["Heavy Ball", "Heavy Ball", "Deflected Subgradient", "Deflected Subgradient"],
    "Funzione": ["Matyas", "Himmelblau", "Matyas", "Himmelblau"],
    "Alpha": [1, 0.01, 1, 0.02],
    "Mu/Gamma": [0.6, 0.1, 0.2, 0.9],
    "Iterazioni": [104, 46, 11, 11],
    "Loss Finale": [0.00000000, 0.00000000, 0.00000000, 0.00000000],
    "Instabilità": [0.00000000, 0.00000000, 0.00000000, 0.00000000],
    "Velocità di Convergenza": [89.7, 95.5, 99.0, 99.0],
    "Tempo di Esecuzione (s)": [0.06, 0.02, 0.00, 0.00]
}
df_summary = pd.DataFrame(data)
df_summary


# In[32]:


#CONFRONTO ALGORITMI HB E DSG NELLA FUNZIONE DI MATYAS
import matplotlib.pyplot as plt

# Definizione della funzione di Matyas
def matyas_function(coords):
    x, y = coords
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

# Generazione della griglia per il plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = matyas_function([X, Y])

# Parametri per il calcolo
x_start, y_start = 4.0, 4.0  # Punto iniziale
num_iterations = 100

# Definizione di figure e assi
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))



# **1. Heavy Ball Method**
model = MatyasFunction()
model.W1 = np.array([x_start, y_start])
alpha, mu = 1, 0.6
losses, descent_path = heavy_ball_method(model, alpha, mu, max_iterations=num_iterations)
x_gr, y_gr = zip(*[np.copy(p) for p in descent_path])

axs[0].contour(X, Y, Z, levels=20)
axs[0].plot(x_gr, y_gr, color='black', markersize=1, label='Heavy Ball Path')
axs[0].set_xlabel('x')
axs[0].set_title('Matyas Function Level Set, Heavy Ball Path')
axs[0].grid(True)

# **2. Deflected Subgradient**
model = MatyasFunction()
model.W1 = np.array([x_start, y_start])
alpha, gamma = 1, 0.2
losses, descent_path = deflected_subgradient_method(model, alpha, gamma, max_iterations=num_iterations)
x_gr, y_gr = zip(*[np.copy(p) for p in descent_path])

axs[1].contour(X, Y, Z, levels=20)
axs[1].plot(x_gr, y_gr, color='black', markersize=1, label='Deflected Subgradient Path')
axs[1].set_xlabel('x')
axs[1].yaxis.set_label_position("right")
axs[1].set_ylabel('y')
axs[1].set_title('Matyas Function Level Set, Deflected Subgradient Path')
axs[1].grid(True)

plt.tight_layout()
plt.show()


# In[33]:


x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Calcolo della funzione di Himmelblau sulla griglia
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

# Parametri per il calcolo
x_start, y_start = 4.0, 4.0  # Punto iniziale
num_iterations = 100

# Definizione di figure e assi
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

# **1. Heavy Ball Method**
model = HimmbleauFunction()
model.W1 = np.array([x_start, y_start])
alpha, mu = 0.01, 0.1
LOSS, descent_path = heavy_ball_method(model, alpha, mu, max_iterations=num_iterations)
x_gr, y_gr = zip(*[np.copy(p) for p in descent_path])  # Estrai i punti per il grafico

axs[0].contour(X, Y, Z, levels=20)
axs[0].plot(x_gr, y_gr, color='black', markersize=1, label='Heavy Ball Path')
axs[0].set_xlabel('x')
axs[0].set_title('Himmelblau Function Level Set, Heavy Ball Path')
axs[0].grid(True)

# **2. Deflected Subgradient**
model = HimmbleauFunction()
model.W1 = np.array([x_start, y_start])
alpha, gamma = 0.02, 0.9
LOSS, descent_path = deflected_subgradient_method(model, alpha, gamma, max_iterations=num_iterations)
x_gr, y_gr = zip(*[np.copy(p) for p in descent_path])  # Estrai i punti per il grafico

axs[1].contour(X, Y, Z, levels=20)
axs[1].plot(x_gr, y_gr, color='black', markersize=1, label='Deflected Subgradient Path')
axs[1].set_xlabel('x')
axs[1].yaxis.set_label_position("right")
axs[1].set_ylabel('y')
axs[1].set_title('Himmelblau Function Level Set, Deflected Subgradient Path')
axs[1].grid(True)

plt.tight_layout()
plt.show()


# In[18]:


import matplotlib.pyplot as plt

# Creazione del plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot per la funzione di Matyas
axes[0].plot(range(len(losses_hb_matyas)), losses_hb_matyas, label="Heavy Ball", linestyle='-', marker='o')
axes[0].plot(range(len(losses_dsg_matyas)), losses_dsg_matyas, label="Deflected Subgradient", linestyle='--', marker='x')
axes[0].set_title("Confronto delle Loss - Funzione di Matyas")
axes[0].set_xlabel("Iterazioni")
axes[0].set_ylabel("Loss")
axes[0].set_yscale("log")  # Scala logaritmica per una migliore leggibilità
axes[0].legend()
axes[0].grid()

# Plot per la funzione di Himmelblau
axes[1].plot(range(len(losses_hb_himmelblau)), losses_hb_himmelblau, label="Heavy Ball", linestyle='-', marker='o')
axes[1].plot(range(len(losses_dsg_himmelbleau)), losses_dsg_himmelbleau, label="Deflected Subgradient", linestyle='--', marker='x')
axes[1].set_title("Confronto delle Loss - Funzione di Himmelblau")
axes[1].set_xlabel("Iterazioni")
axes[1].set_ylabel("Loss")
axes[1].set_yscale("log")  
axes[1].legend()
axes[1].grid()

# Mostra i plot
plt.show()


# In[ ]:




