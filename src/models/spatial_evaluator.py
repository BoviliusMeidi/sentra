import numpy as np
from sklearn.metrics import mean_squared_error
from src.core.base_evaluator import BaseEvaluator

class SpatialEvaluator(BaseEvaluator):
    """
    Implementation for evaluating spatial satellite imagery reconstruction.
    """

    def _compute_ssim(self, y_true: np.ndarray, y_pred: np.ndarray, max_val: float = 10000.0) -> float:
        mu_x, mu_y = np.mean(y_true), np.mean(y_pred)
        var_x, var_y = np.var(y_true), np.var(y_pred)
        cov_xy = np.cov(y_true, y_pred, bias=True)[0][1] if len(y_true) > 1 else 0.0

        c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        return float(numerator/denominator)

    def _compute_r2_explained(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_mean = np.mean(y_true)
        ess = np.sum((y_pred - y_mean) ** 2)
        tss = np.sum((y_true - y_mean) ** 2)

        return float(ess / tss) if tss > 0 else 0.0

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_pred_clipped = np.clip(y_pred, 0.0, 10000.0)
        mse = mean_squared_error(y_true, y_pred_clipped)

        return {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": self._compute_r2_explained(y_true, y_pred_clipped),
            "PSNR": 10 * np.log10((10000.0 ** 2) / mse) if mse > 0 else float("inf"),
            "SSIM": self._compute_ssim(y_true, y_pred_clipped)
        }