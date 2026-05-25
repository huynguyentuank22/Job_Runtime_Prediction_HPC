"""
HDEM_static.py — Static Weighted Ensemble (Ablation of HDEM)

This is an ablation variant of HDEM that removes the dynamic weight update
mechanism (sliding window training). All sub-ensemble weights remain at their
initial uniform values throughout inference, allowing us to isolate and measure
the contribution of the dynamic weight adaptation.

Usage is identical to Dynamic_Weighted_Ensemble, just use Static_Weighted_Ensemble instead.
"""

import numpy as np
from HDEM import Dynamic_Weighted_Ensemble


class Static_Weighted_Ensemble(Dynamic_Weighted_Ensemble):
    """
    Ablation of HDEM: Hierarchical ensemble WITHOUT dynamic weight update.

    The only difference from Dynamic_Weighted_Ensemble is that run_model()
    skips the sliding_window_training() step, so sub-ensemble weights stay
    at their initial uniform values (1/K for K models per sub-ensemble).
    """

    def run_model(self, grid=False):
        # Step 1: Train all base models in every sub-ensemble
        self.training_subensemble()

        # Step 2: Soft regression — weighted average of base model predictions
        pred_subensembles = []
        for i in range(1, self.num_sub + 1):
            sub_ensemble_key = f"sub_ensemble{i}"
            pred = self.soft_regression(
                self.prediction[sub_ensemble_key],
                self.weights[sub_ensemble_key]
            )
            pred_subensembles.append(pred)

        # Step 3: Stack sub-ensemble predictions as meta-features
        meta_X_train = np.column_stack(pred_subensembles)

        # Step 4: Train the meta-learner
        if grid:
            from sklearn.model_selection import RandomizedSearchCV
            param_grids = self.ml.param_grids.get(self.meta_model_name, {})
            model = self.ml.create_model(self.meta_model_name)
            self.meta_model = RandomizedSearchCV(
                model, param_distributions=param_grids,
                n_iter=30, cv=5, scoring='r2',
                verbose=1, random_state=42, n_jobs=-1
            )
            self.meta_model.fit(meta_X_train, self.ml.Y_train)
            print('Best Param ', self.meta_model.best_estimator_)
            print('Best Score ', self.meta_model.best_score_)
        else:
            self.meta_model = self.ml.create_model(self.meta_model_name)
            self.meta_model.fit(meta_X_train, self.ml.Y_train)

        # =====================================================================
        # KEY DIFFERENCE: No sliding_window_training() call here.
        # Weights remain static (uniform) — no dynamic adaptation.
        # =====================================================================

        print('Key on Test Set (Static Weights — No Dynamic Update)')
        return self.model_predict(self.ml.X_test, self.ml.Y_test)
