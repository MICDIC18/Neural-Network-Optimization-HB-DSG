#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Path del file CSV
file_path = r"C:\\Users\\HP\\Downloads\\digits_dataset.csv"

# Lettura del dataset dal file CSV
digits = pd.read_csv(file_path)

# Separazione delle features (X) e del target (y)
X = digits.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
y = digits.iloc[:, -1].values   # L'ultima colonna

print("Dimensioni di X:", X.shape)
print("Dimensioni di y:", y.shape)


# In[2]:


digits


# In[3]:


print(X)


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Risagoma i dati in immagini 8x8 (assumendo che le immagini abbiano questa dimensione)
images = X.reshape(-1, 8, 8)  # Ogni riga di X è una immagine piatta (64 valori)

# Visualizza alcune immagini
fig, axes = plt.subplots(1, 10, figsize=(10, 3))
for ax, image, label in zip(axes, images[:10], y[:10]):  # Mostra le prime 10 immagini e le loro etichette
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'{label}')
plt.show()


# In[5]:


image = X[0].reshape(8, 8)  # Seleziona la prima immagine

# Crea il plot della griglia dei pixel
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='binary') 

# Aggiungi i valori dei pixel nella griglia
for i in range(8):
    for j in range(8):
        pixel_value = image[i, j]
        color = 'black' if pixel_value < 8 else 'white' 
        plt.text(j, i, int(pixel_value), ha='center', va='center', color=color, fontsize=10)

# Configura il titolo e rimuovi gli assi
plt.title("Griglia dei pixel con livelli di luminosità", fontsize=14)
plt.axis('off')

plt.show()


# In[6]:


#Normalizzazione dati
X=X/16.0
X


# In[7]:


from sklearn.model_selection import train_test_split

# Splittare il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# In[8]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[9]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.fit_transform(y_test.reshape(-1, 1))


# In[10]:


print(y_train.shape)
print(y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:




