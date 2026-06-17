# HDEM: A Hierarchical Dynamic Ensemble Model for Runtime Prediction on HPC Systems

[![Paper](https://img.shields.io/badge/Paper-FDSE_2025-blue.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

This repository contains the official source code for the paper **"HDEM: A Hierarchical Dynamic Ensemble Model for Accurate Runtime Prediction on High Performance Computing Systems"**, published at the International Conference on Future Data and Security Engineering (FDSE) 2025.

## 📖 Overview

Predicting job runtime accurately is crucial for optimizing scheduling and resource allocation in High Performance Computing (HPC) environments. However, HPC workloads are highly dynamic and often experience **concept drift** due to varying job characteristics and system conditions over time. 

**HDEM** addresses these challenges by introducing a **Hierarchical Dynamic Ensemble Model**. Our architecture effectively captures complex patterns and adapts to concept drift dynamically, significantly outperforming existing state-of-the-art baselines.

### 🧠 HDEM Architecture

The HDEM architecture is designed to be robust and highly adaptive:
1. **Base Sub-Ensembles**: Multiple distinct machine learning and deep learning models (e.g., XGBoost, LightGBM, CatBoost, MLPs) are grouped into sub-ensembles to capture diverse workload patterns and features.
2. **Dynamic Weighting Mechanism**: Utilizing a sliding window training approach, the model continuously monitors the performance of its base components. Weights of the base models are updated dynamically using a discount factor based on their predictive accuracy, allowing the model to quickly detect and adapt to concept drift.
3. **Hierarchical Meta-Model**: A higher-level meta-model aggregates the predictions from the dynamically weighted sub-ensembles to produce the final, highly accurate runtime prediction.

## 📊 Datasets Evaluated

We extensively evaluate our model on real-world HPC workloads to prove its generalization and robustness:
- **ANL (Argonne National Laboratory)**: A large-scale dataset representing complex, real-world job executions on massive supercomputers.
- **HCMUT (Ho Chi Minh City University of Technology)**: An institutional HPC dataset that reflects academic and research-oriented workloads.

*(Data preprocessing scripts and related files can be found in the `raw_dataset`, `ANL/`, and `HCMUT/` directories.)*

## 📈 Evaluated Baselines

To demonstrate the superiority of HDEM, we compare it against several strong baselines, including both traditional machine learning models and state-of-the-art deep learning approaches for job runtime prediction:
- **PC-Transformer**: A Transformer-based approach specifically tailored for extracting performance characteristics.
- **JREP**: Job Runtime Estimation using deep learning techniques.
- **LSTM / RNN**: Standard recurrent neural network architectures for sequential modeling.
- **SU**: Scheduling Utility models.
- **Random Forest**: A traditional ensemble baseline for tabular data.

*(Source code for the baselines are provided in `PC_transformer.py`, `JREP.py`, `LSTM.py`, `RNN.py`, and `SU.py`)*.

## 🛠️ Setup & Installation

We recommend using **Conda** to manage your environment and dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Job_Runtime_Prediction_HPC.git
cd Job_Runtime_Prediction_HPC

# 2. Create a new conda environment
conda create -n hdem_env python=3.10 -y

# 3. Activate the environment
conda activate hdem_env

# 4. Install the required dependencies
pip install -r requirements.txt
```

*Note: The `requirements.txt` file already includes the extra index URL to install PyTorch with CUDA 12.1 support. Adjust the CUDA version if your local GPU setup requires it.*

## 🚀 Running the Code

The codebase is modularized and organized by datasets (`ANL/` and `HCMUT/`). Each directory contains Jupyter notebooks to train and evaluate HDEM alongside the baselines interactively.

### 1. Training & Evaluation
Navigate to the dataset folder you want to evaluate (e.g., ANL):
```bash
cd ANL
```
Run Jupyter Notebook or Lab to execute the training notebooks:
```bash
jupyter lab
```
- Open **`HDEM_train.ipynb`** to train and evaluate the complete HDEM model.
- You can also explore ablations like `HDEM_static_train.ipynb` or `HDEM_sub_eval_train.ipynb`.
- Open baseline notebooks like `PC_transformer_train.ipynb`, `JREP_train.ipynb`, or `LSTM_train.ipynb` to run the respective baseline evaluations.

### 2. Core Modules
- **`HDEM.py`**: Contains the core classes and implementation of the Hierarchical Dynamic Ensemble Model, including the `Dynamic_Weighted_Ensemble` class, sliding window training, and dynamic weight updates.
- **`preprocessing.py`**: Contains data processing and normalization utilities.

## 📝 Citation

If you use this code or our model in your research, please cite our paper:

```bibtex
@inproceedings{hai2025hdem,
  title={HDEM: A Hierarchical Dynamic Ensemble Model for Accurate Runtime Prediction on High Performance Computing Systems},
  author={Hai, Thanh Hoang Le and Tuan, Huy Nguyen and Dang, Bao Tran and Thuong, Bao Vo and Dinh, Khoi Phan Tran and Thoai, Nam},
  booktitle={International Conference on Future Data and Security Engineering},
  pages={184--198},
  year={2025},
  organization={Springer}
}
```
