import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=3, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size  # Number of neurons per layer
        self.num_layers = num_layers    # Number of LSTM layers
        
        # Define LSTM layer with batch_first=True for (batch, seq, feature) input format
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Linear layer to map LSTM output to a single predicted value
        self.fc = nn.Linear(hidden_size, 1)
        # ReLU activation function to avoid gradient issues
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Select the output of the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through linear layer and apply ReLU
        out = self.fc(out)
        # out = self.relu(out)
        return out
    
    def train_model(self, model, train_loader, val_loader, epochs, lr, scaler):
        model.to(device)  # Move model to GPU/CPU

        criterion = nn.HuberLoss()  
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer with learning rate
        
        best_val_loss = float('inf')  # Track best validation loss for early stopping
        patience = 5  # Patience for early stopping
        patience_counter = 0  # Counter for early stopping
        
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            train_loss = 0.0
            for inputs, _, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
                targets = targets[:, -1, :]
                
                optimizer.zero_grad()  # Clear gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Compute Huber loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                
                train_loss += loss.item() * inputs.size(0)  # Accumulate loss
            
            train_loss /= len(train_loader.dataset)  # Average training loss
            
            # Evaluate on validation set
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient computation
                for inputs, _, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = targets[:, -1, :]
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)  # Average validation loss

            num_features = len(train_loader.dataset.feature_columns)
            val_rmse, val_mae, val_mse, val_r2 = self.evaluate_model(model, val_loader, scaler, num_features)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(model.state_dict(), 'models/best_lstm_model.pth')
                print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    # Function to make predictions with inverse transform
    def predict(self, model, test_loader, scaler, num_features=6):
        model.eval()  # Set model to evaluation mode
        predictions = []
        with torch.no_grad():
            for inputs, _, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                outputs_np = outputs.cpu().numpy()
                outputs_padded = np.concatenate([np.zeros((outputs_np.shape[0], num_features)), outputs_np], axis=1)
                outputs_inv = scaler.inverse_transform(outputs_padded)[:, -1]
                predictions.append(outputs_inv)

        return np.concatenate(predictions)  # Return concatenated predictions

    # Function to evaluate model on test set with RMSE, MAE, MSE, and R2 metrics
    def evaluate_model(self, model, test_loader, scaler, num_features=6):
        model.eval()  # Set model to evaluation mode
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for inputs, _, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets[:, -1, :]
                outputs = model(inputs)
                
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                outputs_padded = np.concatenate([np.zeros((outputs_np.shape[0], num_features)), outputs_np], axis=1)
                targets_padded = np.concatenate([np.zeros((targets_np.shape[0], num_features)), targets_np], axis=1)
                outputs_inv = scaler.inverse_transform(outputs_padded)[:, -1]
                targets_inv = scaler.inverse_transform(targets_padded)[:, -1]
                predictions.append(outputs_inv)
                true_values.append(targets_inv)
        
        # Concatenate predictions and true values
        predictions = np.concatenate(predictions).flatten()
        true_values = np.concatenate(true_values).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        return rmse, mae, mse, r2