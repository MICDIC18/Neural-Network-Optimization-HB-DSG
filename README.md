# Neural-Network-Optimization-HB-DSG
Comparative study of optimization methods for neural networks — Momentum (Heavy Ball) and Deflected Subgradient — implemented for the Digits dataset as part of the Optimization for Data Science course at the University of Pisa.

This repository contains the implementation, analysis, and visualization of two optimization algorithms applied to the Digits dataset. 
The goal is to compare their convergence behavior, efficiency, and robustness in neural network training.

Repository Structure
| Path | Description |
|------|--------------|
| `data/` | Dataset and preprocessing scripts |
| ├── `digits_dataset.csv` | Original dataset |
| ├── `PREPROCESSING_DATASET.ipynb` | Preprocessing notebook |
| └── `PREPROCESSING_DATASET.py` | Preprocessing script |
| `f-star/` | f* values with different λ (lambda) |
| ├── `f_star_con_lambda=0.01.ipynb` | λ = 0.01 |
| ├── `f_star_con_lambda=0.1.ipynb` | λ = 0.1 |
| └── `f_star_con_lambda=1.ipynb` | λ = 1 |
| `plot/` | Plotting scripts for HB and DSG algorithms |
| ├── `Algoritmi_plot.py` | Main plotting script |
| ├── `Plot_hb_e_dsg.ipynb` | Notebook visualization |
| └── `Plot_hb_e_dsg.py` | Python plotting script |
| `test/` | Core algorithms and test scripts |
| ├── `HeavyBall.py` | Heavy Ball implementation |
| ├── `DeflectedSubgradient.py` | DSG implementation |
| ├── `GridSearch.py` | Hyperparameter tuning |
| ├── `NeuralNetwork.py` | Neural network structure |
| └── `Test_algoritmi_su_Matyas_e_Himmelblau.py` | Benchmark tests |
| `weights_and_bias_generated/` | Saved weights and biases (.npy files) |
| ├── `W1.npy`, `W2.npy` | Weight matrices |
| ├── `b1.npy`, `b2.npy` | Bias vectors |
| `MAIN.py` | Main execution script |
| `O4DS_ArancioFebbo_Dicandia.pdf` | Final project report *(Italian)* |
| `O4DS_ArancioFebbo_Dicandia_ENG.pdf` | Final project report *(English)* |
| `README.md` | Project documentation |







The full theoretical and experimental analysis is presented in the report:
- O4DS_ArancioFebbo_Dicandia.pdf (Italian Version)
- O4DS_ArancioFebbo_Dicandia_ENG.pdf (English Version)




University of Pisa — Optimization for Data Science (O4DS)

