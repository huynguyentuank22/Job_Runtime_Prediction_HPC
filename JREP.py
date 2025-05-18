from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib
import numpy as np

class JREP():
    def __init__(self):
        self.base_learners = [
            ('rf', RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, max_features=None, bootstrap=True)),
            ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30)),
            ('svm', SVR(kernel='rbf', C=2.0, epsilon=0.1, gamma='scale')),
            ('dnn', MLPRegressor(hidden_layer_sizes=(64, 128, 64), activation='relu', solver='adam', max_iter=1000, learning_rate_init=0.001))
        ]
        self.meta_learner = MLPRegressor(hidden_layer_sizes=(64, 128, 32), activation='relu', solver='adam', max_iter=1000, learning_rate_init=0.001)
        self.stacking_model = StackingRegressor(
            estimators=self.base_learners,
            final_estimator=self.meta_learner,
            cv=5,
        )
    
    def fit(self, X_train, Y_train):
        self.stacking_model.fit(X_train, Y_train)
        # Save the model
        joblib.dump(self.stacking_model, 'models/stacking_model.pkl')

    def predict(self, X_test, scaler, num_features):
        Y_pred = self.stacking_model.predict(X_test)
        # Inverse transform the predictions
        Y_pred = scaler.inverse_transform(np.concatenate([np.zeros((Y_pred.shape[0], num_features)), Y_pred.reshape(-1, 1)], axis=1))[:, -1]
        return Y_pred
    
    def evaluate_model(self, Y_test, Y_pred, scaler, num_features):
        # Inverse transform the test data
        Y_test = scaler.inverse_transform(np.concatenate([np.zeros((Y_test.shape[0], num_features)), Y_test.reshape(-1, 1)], axis=1))[:, -1]
        # Calculate metrics
        rmse = root_mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        return rmse, mae, mse, r2
