# Neural-Network-Optimization-HB-DSG
Comparative study of optimization methods for neural networks — Momentum (Heavy Ball) and Deflected Subgradient — implemented for the Digits dataset as part of the Optimization for Data Science course at the University of Pisa.

This repository contains the implementation, analysis, and visualization of two optimization algorithms applied to the Digits dataset. 
The goal is to compare their convergence behavior, efficiency, and robustness in neural network training.

Repository Structure

Neural-Network-Optimization-HB-DSG/
│
├── data/                          # Dataset and preprocessing scripts
│   ├── digits_dataset.csv
│   ├── PREPROCESSING_DATASET.ipynb
│   └── PREPROCESSING_DATASET.py
│
├── f-star/                        # f* values with different lambda
│   ├── f_star_con_lambda=0.01.ipynb
│   ├── f_star_con_lambda=0.1.ipynb
│   └── f_star_con_lambda=1.ipynb
│
├── plot/                          # Plotting scripts for HB and DSG algorithms
│   ├── Algoritmi_plot.py
│   ├── Plot_hb_e_dsg.ipynb
│   └── Plot_hb_e_dsg.py
│
├── test/                          # Core algorithms and test scripts
│   ├── HeavyBall.py
│   ├── DeflectedSubgradient.py
│   ├── GridSearch.py
│   ├── NeuralNetwork.py
│   └── Test_algoritmi_su_Matyas_e_Himmelblau.py
│
├── weights_and_bias_generated/    # Saved weights and biases (.npy files)
│   ├── W1.npy
│   ├── W2.npy
│   ├── b1.npy
│   └── b2.npy
│
├── MAIN.py                        # Main execution script
│
├── O4DS_ArancioFebbo_Dicandia.pdf       # Final project report (Italian)
├── O4DS_ArancioFebbo_Dicandia_ENG.pdf   # Final project report (English)
│
└── README.md







The full theoretical and experimental analysis is presented in the report:
- O4DS_ArancioFebbo_Dicandia.pdf (Italian Version)
- O4DS_ArancioFebbo_Dicandia_ENG.pdf (English Version)




University of Pisa — Optimization for Data Science (O4DS)

