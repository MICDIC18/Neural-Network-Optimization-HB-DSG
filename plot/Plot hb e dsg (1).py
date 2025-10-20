#!/usr/bin/env python
# coding: utf-8

# In[1]:


#CARICAMENTO RETE NEURALE
import sys

sys.path.append(r"C:\\Users\\HP\\Downloads\\progetto optimization\\test")

from NeuralNetwork import NeuralNetwork

sys.path.append(r"C:\\Users\\HP\\Downloads\\progetto optimization\\plot") 

# Importa le funzioni dal file Algoritmi_plot.py
from Algoritmi_plot import heavy_ball_optimization, deflected_subgradient_optimization


# In[2]:


#CARICAMENTO DATASET E PRE-PROCESSING
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Path del file CSV
file_path = r"C:\\Users\\HP\\Downloads\\digits_dataset.csv"

# Lettura del dataset dal file CSV
digits = pd.read_csv(file_path)

# Separazione delle features (X) e del target (y)
X = digits.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
y = digits.iloc[:, -1].values   # L'ultima colonna
X=X/16
# Splittare il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.fit_transform(y_test.reshape(-1, 1))


# In[3]:


weights_path = "C:\\Users\\HP\\Downloads\\progetto optimization\\pesi e bias generati"


# # LAMBDA=0,01

# CONFRONTO TEORICO VS EMPIRICO HEAVY BALL

# Heavy Ball	alpha=0.1	mu=0.9	lambda=0.01	loss=0.981280	conv_speed=43.79	unstability=14.001014	

# In[4]:


nn=NeuralNetwork(input_size=64, hidden_size=32, output_size=10, l1_lambda=0.01)
losses_hb_001,_= heavy_ball_optimization(nn, X_train, y_train, alpha=0.1, mu=0.9, max_iterations=10000)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Stima di f* (ottenuta con 100.000 iterazioni)
f_star_hb_001 = 0.961132

# Gap relativo empirico
relative_gap_empirical_001 = np.array(losses_hb_001) - f_star_hb_001

# Determinare il numero di iterazioni
num_iterations = len(losses_hb_001)

# Valori teorici O(1/k)
k_values_001 = np.arange(1, num_iterations + 1)
relative_gap_theoretical = 1 / k_values_001  # O(1/k) 

# **Plot del confronto teorico vs empirico**
plt.figure(figsize=(8, 6))

plt.loglog(k_values_001, relative_gap_empirical_001, label="Empirico: HB (α=0.1, μ=0.9)", linewidth=2)
plt.loglog(k_values_001, relative_gap_theoretical, linestyle="dashed", label="Teorico: $O(1/k)$", linewidth=2)

plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto Convergence Rate teorico vs HB (α=0.1, μ=0.9): LAMBDA=0.01")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()


# CONFRONTO TEORICO VS EMPIRICO DEFLECTED SUBGRADIENT
Deflected Subgradient	alpha=20	gamma=0.1	lambda=0.01	loss=1.131927	conv_speed=39.78	unstability=0.0
# In[ ]:


nn=NeuralNetwork(input_size=64, hidden_size=32, output_size=10, l1_lambda=0.01)
losses_dsg_001,_001= deflected_subgradient_optimization(nn, X_train, y_train, alpha_init=20, gamma=0.1, max_iterations=10000)


# In[ ]:


f_star_dsg_001=0.961132

# Calcola il gap relativo empirico: f(x_k) - f*
k_values_001 = np.arange(1, len(losses_dsg_001) + 1)
relative_gap_empirical_001 = np.abs(np.array(losses_dsg_001) - f_star_dsg_001)

# Calcola il tasso teorico O(1/sqrt(k)) senza normalizzazione
relative_gap_theoretical = 1 / np.sqrt(k_values_001)

# Plot log-log
plt.figure(figsize=(8,6))
plt.loglog(k_values_001, relative_gap_empirical_001, label="Empirico: DSG (α=20, γ=0.1)", linewidth=2)
plt.loglog(k_values_001, relative_gap_theoretical, linestyle="dashed", label="Teorico: $O(1/\sqrt{k})$", linewidth=2)

# Personalizzazione grafico
plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto Convergence Rate teorico vs DSG (α=20, γ=0.1): LAMBDA=0.01")
plt.legend()
plt.grid(True)
plt.show()


# CONFRONTO EMPIRICO HEAVY BALL VS DEFLECTED SUBGRADIENT

# In[ ]:


# Calcolo del gap relativo empirico
relative_gap_hb_001 = np.array(losses_hb_001) - f_star_hb_001
relative_gap_dsg_001 = np.array(losses_dsg_001) - f_star_dsg_001

# Creazione delle scale per l'asse x (iterazioni)
k_values_hb_001 = np.arange(1, len(losses_hb_001) + 1)
k_values_dsg_001 = np.arange(1, len(losses_dsg_001) + 1)

# **Plot log-log del gap relativo**
plt.figure(figsize=(8, 6))
plt.loglog(k_values_hb_001, relative_gap_hb_001, label="Heavy Ball (α=0.1, μ=0.9)", linewidth=2)
plt.loglog(k_values_dsg_001, relative_gap_dsg_001, label="Deflected Subgradient (α=20, γ=0.1)", linewidth=2)

# Personalizzazione del grafico
plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto velocità di convergenza: HB vs DSG: LAMBDA=0.01")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()


# # LAMBDA=0,1

# CONFRONTO TEORICO VS EMPIRICO HEAVY BALL
Heavy Ball	alpha=0.01	mu=0.5	lambda=0.1	loss=2.419762	convergence_speed=96.76	unstability=25.675273	
# In[ ]:


nn=NeuralNetwork(input_size=64, hidden_size=32, output_size=10, l1_lambda=0.1)
losses_hb_01,_= heavy_ball_optimization(nn, X_train, y_train, alpha=0.01, mu=0.5, max_iterations=10000)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Stima di f* (ottenuta con 100.000 iterazioni)
f_star_hb_01 = 2.30201

# Gap relativo empirico
relative_gap_empirical_01 = np.array(losses_hb_01) - f_star_hb_01

# Determinare il numero di iterazioni
num_iterations = len(losses_hb_01)

# Valori teorici O(1/k)
k_values_01 = np.arange(1, num_iterations + 1)
relative_gap_theoretical = 1 / k_values_01  # O(1/k) 

# **Plot del confronto teorico vs empirico**
plt.figure(figsize=(8, 6))

plt.loglog(k_values_01, relative_gap_empirical_01, label="Empirico: HB (α=0.01, μ=0.5)", linewidth=2)
plt.loglog(k_values_01, relative_gap_theoretical, linestyle="dashed", label="Teorico: $O(1/k)$", linewidth=2)

plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto Convergence Rate teorico vs Heavy Ball (α=0.01, μ=0.5): LAMBDA=0.1")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()


# CONFRONTO TEORICO VS EMPIRICO DEFLECTED SUBGRADIENT
Deflected Subgradient	alpha=2	gamma=0.1	lambda=0.1	loss=2.302606	conv_speed=97.76	unstability=0.534332
# In[ ]:


nn=NeuralNetwork(input_size=64, hidden_size=32, output_size=10, l1_lambda=0.1)
losses_dsg_01,_= deflected_subgradient_optimization(nn, X_train, y_train, alpha_init=2, gamma=0.1, max_iterations=10000)


# In[ ]:


f_star_dsg_01=2.30201

# Calcola il gap relativo empirico: f(x_k) - f*
k_values_01 = np.arange(1, len(losses_dsg_01) + 1)
relative_gap_empirical_01 = np.abs(np.array(losses_dsg_01) - f_star_dsg_01)

# Calcola il tasso teorico O(1/sqrt(k)) 
relative_gap_theoretical = 1 / np.sqrt(k_values_01)

# Plot log-log
plt.figure(figsize=(8,6))
plt.loglog(k_values_01, relative_gap_empirical_01, label="Empirico: DSG (α=2, γ=0.1)", linewidth=2)
plt.loglog(k_values_01, relative_gap_theoretical, linestyle="dashed", label="Teorico: $O(1/\sqrt{k})$", linewidth=2)

# Personalizzazione grafico
plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto Convergence Rate teorico vs DSG (α=2, γ=0.1): LAMBDA=0.1")
plt.legend()
plt.grid(True)
plt.show()


# CONFRONTO EMPIRICO HEAVY BALL VS DEFLECTED SUBGRADIENT

# In[ ]:


# Calcolo del gap relativo empirico
relative_gap_hb_01 = np.array(losses_hb_01) - f_star_hb_01
relative_gap_dsg_01 = np.array(losses_dsg_01) - f_star_dsg_01

# Creazione delle scale per l'asse x (iterazioni)
k_values_hb_01 = np.arange(1, len(losses_hb_01) + 1)
k_values_dsg_01 = np.arange(1, len(losses_dsg_01) + 1)

# **Plot log-log del gap relativo**
plt.figure(figsize=(8, 6))
plt.loglog(k_values_hb_01, relative_gap_hb_01, label="Heavy Ball (α=0.01, μ=0.5)", linewidth=2)
plt.loglog(k_values_dsg_01, relative_gap_dsg_01, label="Deflected Subgradient (α=2, γ=0.1)", linewidth=2)

# Personalizzazione del grafico
plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto velocità di convergenza: HB vs DSG: LAMBDA=0.1")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()


# 
# # LAMBDA=1

# CONFRONTO TEORICO VS EMPIRICO HEAVY BALL

# In[ ]:


Heavy Ball	alpha=0.01	mu=0.5	lambda=1	loss=14.064945	conv_speed60.20	unstability=791.206456	


# In[ ]:


nn=NeuralNetwork(input_size=64, hidden_size=32, output_size=10, l1_lambda=1)
losses_hb_1,_= heavy_ball_optimization(nn, X_train, y_train, alpha=0.01, mu=0.5, max_iterations=10000)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Stima di f* (ottenuta con 30.000 iterazioni)
f_star_hb_1 = 13.35865449

# Gap relativo empirico
relative_gap_empirical_1 = np.array(losses_hb_1) - f_star_hb_1

# Determinare il numero di iterazioni
num_iterations = len(losses_hb_1)

# Valori teorici O(1/k)
k_values_1 = np.arange(1, num_iterations + 1)
relative_gap_theoretical = 1 / k_values_1  # O(1/k) 

# **Plot del confronto teorico vs empirico**
plt.figure(figsize=(8, 6))

plt.loglog(k_values_1, relative_gap_empirical_1, label="Heavy Ball (α=0.01, μ=0.5)", linewidth=2)
plt.loglog(k_values_1, relative_gap_theoretical, linestyle="dashed", label="Teorico: $O(1/k)$", linewidth=2)

plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto Convergence Rate teorico vs  Heavy Ball (α=0.01, μ=0.5): LAMBDA=1")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()


# CONFRONTO TEORICO VS EMPIRICO DEFLECTED SUBGRADIENT
Deflected Subgradient alpha=0.2	gamma=0.1	lambda=1	loss=2.307818	conv_speed=82.70	unstability=5.115300
# In[ ]:


nn=NeuralNetwork(input_size=64, hidden_size=32, output_size=10, l1_lambda=1)
losses_dsg_1,_= deflected_subgradient_optimization(nn, X_train, y_train, alpha_init=0.2, gamma=0.1, max_iterations=10000)


# In[ ]:


f_star_dsg_1=2.302321

# Calcola il gap relativo empirico: f(x_k) - f*
k_values_1 = np.arange(1, len(losses_dsg_1) + 1)
relative_gap_empirical_1 = np.abs(np.array(losses_dsg_1) - f_star_dsg_1)

# Calcola il tasso teorico O(1/sqrt(k)) senza normalizzazione
relative_gap_theoretical = 1 / np.sqrt(k_values_1)

# Plot log-log
plt.figure(figsize=(8,6))
plt.loglog(k_values_1, relative_gap_empirical_1, label="Empirico: DSG (α=0.2, γ=0.1)", linewidth=2)
plt.loglog(k_values_1, relative_gap_theoretical, linestyle="dashed", label="Teorico: $O(1/\sqrt{k})$", linewidth=2)

# Personalizzazione grafico
plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto Convergence Rate: DSG (α=0.2, γ=0.1): LAMBDA=1")
plt.legend()
plt.grid(True)
plt.show()


# CONFRONTO EMPIRICO HEAVY BALL VS DEFLECTED SUBGRADIENT

# In[ ]:


# Calcolo del gap relativo empirico
relative_gap_hb_1 = np.array(losses_hb_1) - f_star_hb_1
relative_gap_dsg_1 = np.array(losses_dsg_1) - f_star_dsg_1

# Creazione delle scale per l'asse x (iterazioni)
k_values_hb_1 = np.arange(1, len(losses_hb_1) + 1)
k_values_dsg_1 = np.arange(1, len(losses_dsg_1) + 1)

# **Plot log-log del gap relativo**
plt.figure(figsize=(8, 6))
plt.loglog(k_values_hb_1, relative_gap_hb_1, label="Heavy Ball (α=0.01, μ=0.5)", linewidth=2)
plt.loglog(k_values_dsg_1, relative_gap_dsg_1, label="Deflected Subgradient (α=0.2, γ=0.1)", linewidth=2)

# Personalizzazione del grafico
plt.xlabel("Iterazioni (k)")
plt.ylabel("Gap relativo $f(x_k) - f^*$")
plt.title("Confronto velocità di convergenza: HB vs DSG: LAMBDA=1")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




