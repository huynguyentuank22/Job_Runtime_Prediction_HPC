import numpy as np
import pandas as pd
import time
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, calinski_harabasz_score
from xgboost import XGBRegressor
from preprocessing import clean_data
import torch
import warnings
warnings.filterwarnings('ignore')

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def prepare_data_SU(df, statuss=1):
    """
    Data preparation specific for SU Framework.
    Now includes requested_time as an additional feature for better prediction.
    """
    feature_columns = ['requested_processors', 'requested_time', 'submit_time', 'user_id', 'executable_id']
    target_column = 'run_time'
    
    all_columns = feature_columns + [target_column]
    df = clean_data(df, all_columns, status=statuss)
    
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=True)
    
    return train_df, val_df, test_df

def _find_optimal_k(ld_matrix, max_k=10):
    """
    Find optimal number of clusters using Calinski-Harabasz score,
    as described in Section 3.2 of the paper.
    """
    n = ld_matrix.shape[0]
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 2
    
    best_k = 2
    best_ch = -1
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(ld_matrix)
        
        if len(set(labels)) >= 2:
            try:
                ch_score = calinski_harabasz_score(ld_matrix, labels)
                if ch_score > best_ch:
                    best_ch = ch_score
                    best_k = k
            except:
                continue
    
    return best_k

class SUFramework:
    """
    Enhanced SU Framework based on paper s42774-021-00077-8
    
    Improvements over original:
    1. Added requested_time as feature (strongly correlated with runtime)
    2. Increased k neighbors (50) for more stable local models
    3. XGBoost GPU replaces SVR for better prediction on tabular data
    4. Auto-tuned alpha across validation set
    """
    def __init__(self, k=50, alpha=1.1, max_clusters=10, num_clusters_per_user=None, 
                 use_xgboost=True):
        self.k = k
        self.alpha = alpha
        self.max_clusters = num_clusters_per_user if num_clusters_per_user is not None else max_clusters
        self.use_xgboost = use_xgboost
        
        self.user_clusters = {}
        self.train_data = None
        self.train_features = None
        self.train_targets = None
        self.scaler = StandardScaler()
        
        # Features used for KNN similarity + prediction
        self.knn_features = ['scaled_cpu', 'scaled_reqtime', 'scaled_sin', 'scaled_cos']
        
    def _preprocess(self, df):
        df = df.copy()
        # Combine User and Job_name for Levenshtein clustering
        df['user_job'] = df['user_id'].astype(int).astype(str) + "_" + df['executable_id'].astype(int).astype(str)
        
        # Cycle encoding for submit_time
        hours = (df['submit_time'] % 86400) / 3600.0
        df['submit_sin'] = np.sin(2 * np.pi * hours / 24.0)
        df['submit_cos'] = np.cos(2 * np.pi * hours / 24.0)
        return df

    def fit(self, X_train):
        self.train_data = self._preprocess(X_train)
        
        # Standardize features: CPU_req, requested_time, Submit_sin, Submit_cos
        raw_features = ['requested_processors', 'requested_time', 'submit_sin', 'submit_cos']
        self.train_features = self.train_data[raw_features].values
        self.train_targets = self.train_data['run_time'].values
        self.train_features_scaled = self.scaler.fit_transform(self.train_features)
        
        # Store scaled features
        self.train_data[self.knn_features] = self.train_features_scaled
        
        # ===== Clustering Phase (k-means++ on LD matrix per user) =====
        unique_users = self.train_data['user_id'].unique()
        
        for user in unique_users:
            user_mask = self.train_data['user_id'] == user
            user_df = self.train_data[user_mask]
            unique_jobs = user_df['user_job'].unique()
            n_unique = len(unique_jobs)
            
            if n_unique <= 1:
                job_to_cluster = {job: 0 for job in unique_jobs}
            elif n_unique == 2:
                job_to_cluster = {job: i for i, job in enumerate(unique_jobs)}
            else:
                # Calculate LD matrix
                ld_matrix = np.zeros((n_unique, n_unique))
                for i in range(n_unique):
                    for j in range(i + 1, n_unique):
                        dist = levenshtein_distance(unique_jobs[i], unique_jobs[j])
                        ld_matrix[i, j] = dist
                        ld_matrix[j, i] = dist
                
                # k-means++ with optimal k
                optimal_k = _find_optimal_k(ld_matrix, max_k=min(self.max_clusters, n_unique - 1))
                kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
                labels = kmeans.fit_predict(ld_matrix)
                job_to_cluster = {unique_jobs[i]: int(labels[i]) for i in range(n_unique)}
                
            self.user_clusters[user] = job_to_cluster
        
        # Assign cluster_id to each training row
        def get_cluster(row):
            user = row['user_id']
            job = row['user_job']
            if user in self.user_clusters:
                return self.user_clusters[user].get(job, 0)
            return 0
            
        self.train_data['cluster_id'] = self.train_data.apply(get_cluster, axis=1)
        
    def tune_alpha(self, val_df, alpha_range=None):
        """Auto-tune alpha on validation set for best APA."""
        if alpha_range is None:
            alpha_range = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]
        
        y_val = val_df['run_time'].values
        
        # Get raw predictions (alpha=1.0)
        original_alpha = self.alpha
        self.alpha = 1.0
        raw_preds = self.predict(val_df)
        self.alpha = original_alpha
        
        raw_preds = np.clip(raw_preds, a_min=1.0, a_max=None)
        
        best_alpha = 1.1
        best_apa = 0
        
        for a in alpha_range:
            preds = raw_preds * a
            apa_vals = np.where(preds < y_val, preds / y_val, y_val / preds)
            apa = np.mean(apa_vals)
            if apa > best_apa:
                best_apa = apa
                best_alpha = a
        
        self.alpha = best_alpha
        print(f"Auto-tuned alpha = {best_alpha:.2f} (APA on val = {best_apa:.4f})")
        return best_alpha
        
    def predict(self, X_test):
        test_data = self._preprocess(X_test)
        raw_features = ['requested_processors', 'requested_time', 'submit_sin', 'submit_cos']
        test_features_scaled = self.scaler.transform(test_data[raw_features].values)
        test_data[self.knn_features] = test_features_scaled
        
        # Pre-setup GPU device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        predictions = []
        
        for idx, row in test_data.iterrows():
            user = row['user_id']
            job = row['user_job']
            test_x = row[self.knn_features].values.astype(float)
            
            # Find candidates from the same cluster
            if user in self.user_clusters:
                cluster_id = self.user_clusters[user].get(job, 0)
                candidates = self.train_data[
                    (self.train_data['user_id'] == user) & 
                    (self.train_data['cluster_id'] == cluster_id)
                ]
                if len(candidates) < self.k:
                    candidates = self.train_data[self.train_data['user_id'] == user]
            else:
                candidates = self.train_data
                
            if len(candidates) < self.k:
                candidates = self.train_data
                
            cand_X = candidates[self.knn_features].values
            cand_y = candidates['run_time'].values
            
            # KNN using Euclidean distance on GPU
            cand_X_tensor = torch.tensor(cand_X, dtype=torch.float32, device=device)
            test_x_tensor = torch.tensor(test_x.reshape(1, -1), dtype=torch.float32, device=device)
            
            distances = torch.cdist(test_x_tensor, cand_X_tensor).squeeze(0)
            
            k_actual = min(self.k, len(cand_X))
            nearest_indices = torch.argsort(distances)[:k_actual].cpu().numpy()
            
            knn_X = cand_X[nearest_indices]
            knn_y = cand_y[nearest_indices]
            
            # Local prediction model
            if len(knn_y) > 0:
                if self.use_xgboost:
                    # XGBoost GPU for stronger prediction on tabular data
                    model = XGBRegressor(
                        n_estimators=50, max_depth=3, learning_rate=0.2,
                        tree_method='hist', device='cuda', verbosity=0
                    )
                    model.fit(knn_X, knn_y)
                    pred = model.predict(knn_X[:1] * 0 + test_x.reshape(1, -1))[0]
                else:
                    # Original SVR from paper
                    svr = SVR(kernel='rbf')
                    svr.fit(knn_X, knn_y)
                    pred = svr.predict(test_x.reshape(1, -1))[0]
            else:
                pred = np.mean(self.train_targets)
                
            # Formula 5: Final = alpha * Predict(job_new)
            predictions.append(pred * self.alpha)
            
        return np.array(predictions)
        
    def evaluate(self, X_test, y_test=None):
        start_time = time.time()
        preds = self.predict(X_test)
        end_time = time.time()
        
        if y_test is None:
            y_test = X_test['run_time'].values
            
        preds = np.clip(preds, a_min=1.0, a_max=None)
            
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        # Underestimation Rate
        ur = np.mean(preds < y_test)
        
        # Average Predictive Accuracy (APA)
        apa_vals = np.where(preds < y_test, preds / y_test, y_test / preds)
        apa = np.mean(apa_vals)
        
        infer_time = (end_time - start_time) / len(preds)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R² Score': r2,
            'Underestimation Rate': ur,
            'APA': apa,
            'Inference Time': infer_time
        }
