#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Classe della rete neurale
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, l1_lambda, seed=42):
        # Fissiamo il seed per riproducibilità
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1_lambda = l1_lambda
        
        # Inizializzazione dei pesi e bias
        # He Initialization per ReLU (std = sqrt(2 / fan_in))
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Inizializzazione simile a Glorot/He per lo strato Softmax
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        dx = np.zeros(x.shape)
        dx[x > 0] = 1
        return dx
    
    def softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(x_shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def backward_propagation(self, X, y):
        dZ2 = self.A2 - y
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.Z1)
        dW2 = (np.dot(self.A1.T, dZ2) / len(y)) + self.l1_lambda * np.sign(self.W2)
        db2 = np.sum(dZ2, axis=0) / len(y)
        dW1 = (np.dot(X.T, dZ1) / len(y)) + self.l1_lambda * np.sign(self.W1)
        db1 = np.sum(dZ1, axis=0) / len(y)
        return dW1, db1, dW2, db2
    
    def evaluate(self, X, y):
        y_prob = self.forward_propagation(X)
        cross_entropy = -np.mean(np.sum(y * np.log(y_prob + 1e-12), axis=1))
        loss_regularization = np.sum(np.abs(self.W1)) + np.sum(np.abs(self.W2))
        total_loss = cross_entropy + self.l1_lambda * loss_regularization
        return total_loss
    
    def gradient(self, X, y):
        return self.backward_propagation(X, y)

