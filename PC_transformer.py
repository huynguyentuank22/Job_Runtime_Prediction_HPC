import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

# Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_trend = nn.Linear(1, dim)
        self.linear_period = nn.Linear(1, dim)

    def forward(self, t):
        trend = F.relu(self.linear_trend(t))
        period = torch.sin(self.linear_period(t))
        return trend + period
    
# Multi-Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)
    
# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
    
# PC-Transformer Block
class PCTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(PCTransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(self.dropout(ff_output))
        return x
    
# Full PC-Transformer Model
class PCTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, output_dim, dropout=0.1):
        super(PCTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.time_embedding = TimeEmbedding(d_model)
        self.transformer_blocks = nn.ModuleList([
            PCTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        x = self.input_linear(x)
        time_emb = self.time_embedding(t)
        x = x + time_emb
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.output_linear(x)
        return x
    
    # Training setup
    def train_model(self, model, dataloaders, num_epochs, device, scaler):
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        patience, trials = 5, 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            for batch_x, batch_t, batch_y in dataloaders['train']:
                batch_x, batch_t, batch_y = batch_x.to(device), batch_t.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_x, batch_t)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(dataloaders['train'])
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, batch_t, batch_y in dataloaders['val']:
                    batch_x, batch_t, batch_y = batch_x.to(device), batch_t.to(device), batch_y.to(device)
                    output = model(batch_x, batch_t)
                    loss = criterion(output, batch_y)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(dataloaders['val'])
            
            num_features = len(dataloaders['train'].dataset.feature_columns)
            # Evaluate metrics on validation set
            val_rmse, val_mae, val_mse, val_r2 = self.evaluate_model(model, dataloaders['val'], device, scaler, num_features)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}')
            
            # Early stopping based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trials = 0
                # Save the best model
                torch.save(model.state_dict(), 'models/best_pc_transformer_model.pth')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping triggered after epoch {epoch+1}')
                    break

    # Evaluation Function for R2 and MAE
    def evaluate_model(self, model, dataloader, device, scaler, num_features=6):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_t, batch_y in dataloader:
                batch_x, batch_t, batch_y = batch_x.to(device), batch_t.to(device), batch_y.to(device)
                output = model(batch_x, batch_t)
                
                # Inverse transform the predictions and targets
                output = output.cpu().numpy().reshape(-1, 1)
                batch_y = batch_y.cpu().numpy().reshape(-1, 1)
                
                # Inverse transform the normalized values
                output = scaler.inverse_transform(np.concatenate([np.zeros((output.shape[0], num_features)), output], axis=1))[:, -1]
                batch_y = scaler.inverse_transform(np.concatenate([np.zeros((batch_y.shape[0], num_features)), batch_y], axis=1))[:, -1]
                
                all_preds.extend(output)
                all_targets.extend(batch_y)
        
        rmse = root_mean_squared_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        return rmse, mae, mse, r2
    
    # Prediction Function
    def predict(self, x, t, device, scaler, num_features=6):
        self.eval()
        x, t = x.to(device), t.to(device)
        
        with torch.no_grad():
            output = self(x, t)
            output = output.cpu().numpy().reshape(-1, 1)
            # Inverse transform the predictions
            output = scaler.inverse_transform(np.concatenate([np.zeros((output.shape[0], num_features)), output], axis=1))[:, -1]
        
        return output