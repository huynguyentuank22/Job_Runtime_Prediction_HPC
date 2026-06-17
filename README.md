# HDEM: A Hierarchical Dynamic Ensemble Model for Runtime Prediction on HPC Systems

[![Paper](https://img.shields.io/badge/Paper-FDSE_2025-blue.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

This repository contains the official source code for the paper **"HDEM: A Hierarchical Dynamic Ensemble Model for Accurate Runtime Prediction on High Performance Computing Systems"**, published at the International Conference on Future Data and Security Engineering (FDSE) 2025.

## Overview

Predicting job runtime accurately is crucial for optimizing scheduling and resource allocation in High Performance Computing (HPC) environments. However, HPC workloads are inherently dynamic and frequently experience **concept drift** due to fluctuating job characteristics and system conditions over time. 

**HDEM** addresses these critical challenges by introducing a **Hierarchical Dynamic Ensemble Model**. Our architecture effectively captures complex workload patterns and dynamically adapts to concept drift, significantly outperforming existing state-of-the-art baselines in predictive accuracy.

### HDEM Architecture

<p align="center">
  <img src="images/something-Page-16.drawio.png" alt="HDEM Architecture Pipeline">
  <br>
  <em>Figure 1: The overall pipeline and architecture of the Hierarchical Dynamic Ensemble Model (HDEM).</em>
</p>

The HDEM architecture is designed to be highly robust and adaptive:
1. **Base Sub-Ensembles**: Multiple distinct machine learning and deep learning models (e.g., XGBoost, LightGBM, CatBoost, MLPs) are strategically grouped into sub-ensembles to capture diverse workload features.
2. **Dynamic Weighting Mechanism**: Utilizing a sliding window training approach, the model continuously monitors the performance of its constituent base models. The weights of the base models are updated dynamically using a discount factor based on their predictive accuracy, enabling the system to promptly detect and adapt to concept drift.
3. **Hierarchical Meta-Model**: A higher-level meta-model aggregates the predictions from the dynamically weighted sub-ensembles to produce the final, highly precise runtime prediction.

<p align="center">
  <img src="images/dynamic_weighting_ablation_2.png" alt="Dynamic Weighting Mechanism">
  <br>
  <em>Figure 2: The effect of the dynamic weighting mechanism. This mechanism allows the model to continuously adapt to concept drift in HPC workloads, sustaining high accuracy over time compared to static approaches.</em>
</p>

## Datasets Evaluated

We rigorously evaluate our model on real-world HPC workloads to demonstrate its generalization capabilities and robustness:
- **ANL (Argonne National Laboratory)**: A large-scale dataset representing complex, real-world job executions on massive supercomputing clusters.
- **HCMUT (Ho Chi Minh City University of Technology)**: An institutional HPC dataset that reflects diverse academic and research-oriented workloads.

*(Data preprocessing scripts and related artifacts can be found in the `raw_dataset`, `ANL/`, and `HCMUT/` directories.)*

## Evaluated Baselines

To establish the efficacy of HDEM, we compare its performance against several prominent baselines, encompassing both traditional machine learning models and state-of-the-art deep learning approaches for job runtime prediction:

<p align="center">
  <img src="images/hpcbert.png" alt="HPCBERT+ Pipeline">
  <br>
  <em>Figure 3: The pipeline of HPCBERT+, a strong Transformer-based baseline utilized for comparative evaluation.</em>
</p>

- **PC-Transformer / HPCBERT+**: A Transformer-based approach specifically tailored for extracting performance characteristics.
- **JREP**: Job Runtime Estimation utilizing deep learning techniques.
- **LSTM / RNN**: Standard recurrent neural network architectures for sequential modeling.
- **SU**: Scheduling Utility models.
- **Random Forest**: A traditional ensemble baseline for tabular data evaluation.

*(Source code for the baseline models is provided in `PC_transformer.py`, `JREP.py`, `LSTM.py`, `RNN.py`, and `SU.py`.)*

## Experimental Results

Our comprehensive evaluation demonstrates the robust predictive capability of HDEM across various metrics and scenarios. 

<p align="center">
  <img src="images/HDEM_Single.png" alt="HDEM Results">
  <br>
  <em>Figure 4: Comparative evaluation results showcasing the superiority of HDEM in runtime prediction accuracy over existing baseline models.</em>
</p>

## Setup and Installation

We recommend using **Conda** to manage the Python environment and dependencies.

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

*Note: The `requirements.txt` file already includes the extra index URL to install PyTorch with CUDA 12.1 support. Adjust the CUDA version if your local hardware setup requires a different configuration.*

## Running the Code

The codebase is modularized and structured by dataset (`ANL/` and `HCMUT/`). Each directory contains comprehensive Jupyter notebooks to train and evaluate HDEM alongside the baselines.

### 1. Training and Evaluation
Navigate to the target dataset directory (e.g., ANL):
```bash
cd ANL
```
Launch Jupyter Notebook or JupyterLab to execute the training environments:
```bash
jupyter lab
```
- Open **`HDEM_train.ipynb`** to train and evaluate the complete HDEM architecture.
- You can explore ablation studies by opening `HDEM_static_train.ipynb` or `HDEM_sub_eval_train.ipynb`.
- Open baseline notebooks (e.g., `PC_transformer_train.ipynb`, `JREP_train.ipynb`, or `LSTM_train.ipynb`) to reproduce the respective baseline evaluations.

### 2. Core Modules
- **`HDEM.py`**: Contains the core class definitions and implementation of the Hierarchical Dynamic Ensemble Model, including the `Dynamic_Weighted_Ensemble` class, sliding window training procedures, and dynamic weight update logic.
- **`preprocessing.py`**: Contains rigorous data processing and normalization utilities.

## Citation

If you utilize this codebase or our model in your research, we kindly request that you cite our paper:

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
