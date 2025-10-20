#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#iMPORTAZIONE FUNZIONI E DATASET PRE-PROCESSATO
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

weights_path = "C:\\Users\\HP\\Downloads\\progetto optimization\\pesi e bias generati"
sys.path.append(r"C:\\Users\\HP\\Downloads\\progetto optimization\\test")

from NeuralNetwork import NeuralNetwork
from HeavyBall import heavy_ball_optimization
from DeflectedSubgradient import deflected_subgradient_optimization
from GridSearch import grid_search_nn


# Path del file CSV
file_path = r"C:\\Users\\HP\\Downloads\\digits_dataset.csv"

# Lettura del dataset dal file CSV
digits = pd.read_csv(file_path)

# Separazione delle features (X) e del target (y)
X = digits.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
y = digits.iloc[:, -1].values   # L'ultima colonna

X=X/16.0

# Splittare il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.fit_transform(y_test.reshape(-1, 1))


# # TEST RETE NEURALE

# LAMBDA=0,01

# In[ ]:


# Definire i parametri da testare
alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]
mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per HB
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per DS
lambda_values = [0.01]  # Valori di regolarizzazione L1


# In[ ]:


# Grid search per Heavy Ball
top_hb = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="HB")


# In[ ]:


top_hb


# In[ ]:


alpha_values = [1, 2, 5, 10, 15, 20, 40]
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per DS


# In[ ]:


# Grid search per Deflected Subgradient
top_dsg = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="DSG")


# In[ ]:


top_dsg


# LAMBDA=0,01 CON 10 MILA ITERAZIONI

# In[ ]:


# Definire i parametri da testare
alpha_values = [0.1, 0.2]
mu_values = [0.7, 0.9]  # Solo per HB
lambda_values = [0.01]  # Valori di regolarizzazione L1


# In[ ]:


# Grid search per Heavy Ball
top_hb = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="HB", max_iterations=10000, epsilon=1e-6)


# In[ ]:


top_hb


# In[ ]:


#Heavy Ball	alpha=0.1	mu=0.9	lambda=0.01	loss=0.981280	conv_speed=43.79	unstability=14.001014	


# In[ ]:


# Definire i parametri da testare
alpha_values = [10, 15, 20]
gamma_values = [0.1, 0.3]  # Solo per DSG
lambda_values = [0.01]  # Valori di regolarizzazione L1


# In[ ]:


# Grid search per Deflected Subgradient
top_dsg = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="DSG", max_iterations=10000, epsilon=1e-6)


# In[ ]:


top_dsg


# In[ ]:


#Deflected Subgradient	alpha=20	gamma=0.1	lambda=0.01	loss=1.131927	conv_speed=39.78	unstability=0.0


# # LAMBDA=0,1

# In[ ]:


# Definire i parametri da testare
alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]
mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per HB
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per DSG
lambda_values = [0.1]  # Valori di regolarizzazione L1


# In[ ]:


# Grid search per Heavy Ball
top_hb = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="HB")


# In[ ]:


top_hb


# In[ ]:


lambda_values = [0.1]  # Valori di regolarizzazione L1
alpha_values = [1, 2, 5, 10, 15, 20, 40]
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per DSG


# In[ ]:


# Grid search per Deflected Subgradient
top_dsg = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="DSG")


# In[ ]:


top_dsg


# LAMBDA=0,1 CON 10 MILA ITERAZIONI

# In[ ]:


# Definire i parametri da testare
alpha_values = [0.01]
mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per HB
lambda_values = [0.1]  # Valori di regolarizzazione L1


# In[ ]:


# Grid search per Heavy Ball
top_hb = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="HB", max_iterations=10000, epsilon=1e-6)


# In[ ]:


top_hb


# In[ ]:


#Heavy Ball	alpha=0.01	mu=0.5	lambda=0.1	loss=2.419762	convergence_speed=96.76	unstability=25.675273	


# In[ ]:


lambda_values = [0.1]  # Valori di regolarizzazione L1
alpha_values = [1, 2, 5]
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per DSG


# In[ ]:


# Grid search per Deflected Subgradient
top_dsg = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="DSG", max_iterations=10000, epsilon=1e-6)


# In[ ]:


top_dsg


# In[ ]:


#Deflected Subgradient	alpha=2	gamma=0.1	lambda=0.1	loss=2.302606	conv_speed=97.76	unstability=0.534332


# # LAMBDA=1

# In[ ]:


# Definire i parametri da testare
alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]
mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per HB
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per DSG
lambda_values = [1]  # Valori di regolarizzazione L1


# In[ ]:


# Grid search per Heavy Ball
top_hb = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="HB")


# In[ ]:


top_hb


# In[ ]:


# Grid search per Deflected Subgradient
top_dsg = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="DSG")


# In[ ]:


top_dsg


# LAMBDA=1 CON 10 MILA ITERAZIONI

# In[ ]:


# Definire i parametri da testare
alpha_values = [0.01, 0.05]
mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per HB
lambda_values = [1]  # Valori di regolarizzazione L1


# In[ ]:


# Grid search per Heavy Ball
top_hb = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="HB", max_iterations=10000, epsilon=1e-6)


# In[ ]:


top_hb


# In[ ]:


#Heavy Ball	alpha=0.01	mu=0.5	lambda=1	loss=14.064945	conv_speed60.20	unstability=791.206456	


# In[ ]:


# Definire i parametri da testare
alpha_values = [0.1, 0.2]
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per DSG
lambda_values = [1]  # Valori di regolarizzazione L1
mu_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Solo per HB


# In[ ]:


# Grid search per Deflected Subgradient
top_dsg = grid_search_nn(NeuralNetwork, X_train, y_train, alpha_values, mu_values, gamma_values, lambda_values, method="DSG", max_iterations=10000, epsilon=1e-6)


# In[ ]:


top_dsg


# In[ ]:


#Deflected Subgradientalpha=0.2	gamma=0.1	lambda=1	loss=2.307818	conv_speed=82.70	unstability=5.115300


# In[ ]:





# In[ ]:





# In[ ]:




