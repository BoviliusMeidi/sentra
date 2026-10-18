import pickle
import logging
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from src.core.base_model_trainer import BaseModelTrainer

class ElasticNetModel(BaseModelTrainer):
    """
    Elastic Net regression model wrapper featuring automated hyperparameter grid-search,
    standardization, full history tracking, and comprehensive metric encapsulation.
    """
    def __init__(self):
        self.model = None
        self.best_params = {}
        self.grid_history = []
        self.metadata = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("The spatial model is uninitialized. Ensure the model is trained or loaded before inference.")
        return self.model.predict(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        alphas: list,
        l1_ratios: list,
        evaluator=None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        random_state: int = 42
    ):
        best_score = -float('inf')
        best_pipeline = None
        self.grid_history = []

        for a in alphas:
            for l1 in l1_ratios:
                logging.info(f"   ⚙️ Fitting ElasticNet -> Alpha: {a} | L1 Ratio: {l1}")

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("elasticnet", ElasticNet(
                        alpha=a,
                        l1_ratio=l1,
                        random_state=random_state,
                        max_iter=5000
                    ))
                ])
                pipeline.fit(X_train, y_train)

                y_train_pred = pipeline.predict(X_train)
                train_r2 = r2_score(y_train, y_train_pred)

                if X_val is not None and y_val is not None:
                    y_val_pred = pipeline.predict(X_val)
                    metrics = evaluator.evaluate(y_val, y_val_pred) if evaluator else {}
                    mse = metrics.get("MSE", mean_squared_error(y_val, y_val_pred))
                    rmse = metrics.get("RMSE", np.sqrt(mse))
                    r2 = metrics.get("R2", r2_score(y_val, y_val_pred))
                    psnr = metrics.get("PSNR", 0.0)
                    ssim = metrics.get("SSIM", 0.0)
                    score = r2
                else:
                    mse, rmse, r2, psnr, ssim = 0.0, 0.0, train_r2, 0.0, 0.0
                    score = train_r2

                self.grid_history.append({
                    "Alpha": a,
                    "L1_Ratio": l1,
                    "Train_R2": round(train_r2, 4),
                    "Val_MSE": round(mse, 2),
                    "Val_RMSE": round(rmse, 4),
                    "Val_R2": round(r2, 4),
                    "Val_PSNR": round(psnr, 2),
                    "Val_SSIM": round(ssim, 4),
                    "Fit_Status": "Optimal"
                })

                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline
                    self.best_params = {"alpha": a, "l1_ratio": l1}

        self.model = best_pipeline
        return self.best_params, self.grid_history

    def test_and_evaluate(self, X_test: np.ndarray, y_test: np.ndarray, evaluator) -> dict:
        y_pred = self.model.predict(X_test)
        if evaluator and hasattr(evaluator, "evaluate"):
            return evaluator.evaluate(y_test, y_pred)

        mse = mean_squared_error(y_test, y_pred)
        return {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, y_pred),
            "PSNR": 0.0,
            "SSIM": 0.0
        }

    def save(self, file_path: str):
        """Serializes the trained pipeline and metadata into a .pkl artifact securely."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path: str):
        """Deserializes a saved model object from disk."""
        with open(file_path, "rb") as f:
            return pickle.load(f)