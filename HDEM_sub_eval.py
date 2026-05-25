"""
HDEM_sub_eval.py — Sub-Ensemble Performance Evaluation for HDEM

Extends Dynamic_Weighted_Ensemble to evaluate each sub-ensemble individually
as an ablation study, measuring how each sub-ensemble contributes to the
final HDEM prediction.
"""

import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from HDEM import Dynamic_Weighted_Ensemble


class SubEnsembleEvaluator(Dynamic_Weighted_Ensemble):
    """
    Extends HDEM to evaluate each sub-ensemble's performance individually.

    After training, call evaluate_subensembles() to get metrics for each
    sub-ensemble's soft regression output evaluated directly against the
    test set (without the meta-learner).
    """

    def evaluate_subensembles(self, X_test=None, y_test=None):
        """
        Evaluate each sub-ensemble individually on the test set.

        Returns:
            dict: {
                'sub_ensemble1': {'MAE': ..., 'RMSE': ..., 'R² Score': ..., 'Inference Time': ..., 'Models': [...]},
                'sub_ensemble2': {...},
                'sub_ensemble3': {...},
            }
        """
        if X_test is None:
            X_test = self.ml.X_test
        if y_test is None:
            y_test = self.ml.Y_test

        results = {}

        for i in range(1, self.num_sub + 1):
            sub_ensemble_key = f"sub_ensemble{i}"

            start_time = time.time()

            # Get predictions from each base model in this sub-ensemble
            sub_preds = {}
            for name, model in self.models[sub_ensemble_key].items():
                sub_preds[name] = model.predict(X_test)

            # Soft regression: weighted average of base model predictions
            pred = self.soft_regression(sub_preds, self.weights[sub_ensemble_key])

            end_time = time.time()

            # Inverse transform predictions and actual values
            pred_reshaped = pred.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)

            dummy_pred = np.zeros((pred_reshaped.shape[0], self.scaler.n_features_in_))
            dummy_test = np.zeros((y_test_reshaped.shape[0], self.scaler.n_features_in_))

            target_idx = -1
            dummy_pred[:, target_idx] = pred_reshaped.flatten()
            dummy_test[:, target_idx] = y_test_reshaped.flatten()

            pred_inv = self.scaler.inverse_transform(dummy_pred)[:, target_idx]
            y_test_inv = self.scaler.inverse_transform(dummy_test)[:, target_idx]

            # Compute metrics
            mae = mean_absolute_error(y_test_inv, pred_inv)
            rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
            r2 = r2_score(y_test_inv, pred_inv)
            infer_time = (end_time - start_time) / len(pred_inv)

            model_names = list(self.models[sub_ensemble_key].keys())
            results[sub_ensemble_key] = {
                'Models': model_names,
                'MAE': mae,
                'RMSE': rmse,
                'R² Score': r2,
                'Inference Time': infer_time
            }

            print(f"\n{sub_ensemble_key} ({', '.join(model_names)}):")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  Inference Time: {infer_time:.6f}s/sample")

        return results

    def run_model_with_sub_eval(self, grid=False):
        """
        Run full HDEM training, then also evaluate each sub-ensemble individually.

        Returns:
            tuple: (hdem_result, sub_results)
                - hdem_result: dict with overall HDEM metrics
                - sub_results: dict with per-sub-ensemble metrics
        """
        # Run the standard HDEM pipeline (with dynamic weights)
        hdem_result = self.run_model(grid=grid)

        # Evaluate each sub-ensemble individually
        print(f"\n{'='*60}")
        print("Sub-Ensemble Individual Performance")
        print(f"{'='*60}")
        sub_results = self.evaluate_subensembles()

        # Print comparison summary
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Model':<45} {'RMSE':>12} {'MAE':>12} {'R²':>10}")
        print("-" * 79)
        for key, val in sub_results.items():
            label = f"{key} ({', '.join(val['Models'])})"
            print(f"{label:<45} {val['RMSE']:>12.4f} {val['MAE']:>12.4f} {val['R² Score']:>10.4f}")
        print("-" * 79)
        print(f"{'HDEM (Full Ensemble)':<45} {hdem_result['RMSE']:>12.4f} {hdem_result['MAE']:>12.4f} {hdem_result['R² Score']:>10.4f}")

        return hdem_result, sub_results
