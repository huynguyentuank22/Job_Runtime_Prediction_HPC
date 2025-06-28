import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import random

class HPCDataset(Dataset):
    def __init__(self, sequences, feature_columns, target_column):
        self.sequences = sequences
        self.feature_columns = feature_columns
        self.target_column = target_column
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        X = seq[self.feature_columns].values
        Y = seq[self.target_column].values.reshape(-1, 1)
        # t = np.arange(len(seq)).reshape(-1, 1)
        t = seq['submit_time'].values.reshape(-1, 1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(t, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_data(df, columns, status=1):
    # Filter out rows with status != 1
    df = df[df['status'] == status]

    # Keep only the specified columns
    df = df[columns]

    # Remove noise: runtime < 600s or runtime < 1% of requested time
    df = df[df['run_time'] >= 600]
    df = df[df['run_time'] >= 0.01 * df['requested_time']]
    
    return df

def normalize_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def sample_data(df, seq_len=20):
    df = df.sort_values('submit_time').reset_index(drop=True)
    sequences = []
    for i in range(0, len(df) - seq_len + 1):
        seq = df.iloc[i:i + seq_len]
        sequences.append(seq)
    return sequences

def prepare_data(df, feature_columns, target_column, statuss=1):
    all_columns = feature_columns + [target_column]

    # Step 1: Clean data
    df = clean_data(df, all_columns, status=statuss)

    # Step 2: Split data into train, test (8:2)
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)

    # Step 3: Normalize train
    train_df, scaler = normalize_data(train_df, all_columns)
    test_df[all_columns] = scaler.transform(test_df[all_columns])

    X_train = train_df[feature_columns].values
    Y_train = train_df[target_column].values
    X_test = test_df[feature_columns].values
    Y_test = test_df[target_column].values

    return X_train, X_test, Y_train, Y_test, scaler

def prepare_data_seq(df, feature_columns, target_column, statuss=1, seq_len=20, batch_size=128):
    # Step 1: Clean data
    all_columns = feature_columns + [target_column] 
    df = clean_data(df, all_columns, status=statuss)

    # Step 2: Split data into train, val, test (8:1:1)
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=True)

    # Step 3: Normalize train
    train_df, scaler = normalize_data(train_df, all_columns)
    val_df[all_columns] = scaler.transform(val_df[all_columns])
    test_df[all_columns] = scaler.transform(test_df[all_columns])

    # Step 4: Sample sequences
    train_sequences = sample_data(train_df, seq_len)
    val_sequences = sample_data(val_df, seq_len)
    test_sequences = sample_data(test_df, seq_len)

    # Step 5: Create datasets
    train_dataset = HPCDataset(train_sequences, feature_columns, target_column)
    val_dataset = HPCDataset(val_sequences, feature_columns, target_column)
    test_dataset = HPCDataset(test_sequences, feature_columns, target_column)

    # Step 6: Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}, scaler

def prepare_data_DL(df, feature_columns, target_column, statuss=1):
    # Step 1: Clean data
    all_columns = feature_columns + [target_column]
    df = clean_data(df, all_columns, status=statuss)

    # Step 2: Split data into train, val, test (8:1:1)
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=True)

    # Step 3: Normalize train
    train_df, scaler = normalize_data(train_df, all_columns)
    val_df[all_columns] = scaler.transform(val_df[all_columns])
    test_df[all_columns] = scaler.transform(test_df[all_columns])

    X_train = train_df[feature_columns].values
    Y_train = train_df[target_column].values
    X_val = val_df[feature_columns].values
    Y_val = val_df[target_column].values
    X_test = test_df[feature_columns].values
    Y_test = test_df[target_column].values

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler