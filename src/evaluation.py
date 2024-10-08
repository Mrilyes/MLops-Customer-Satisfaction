import logging
from abc import ABC , abstractmethod
import numpy as np

from sklearn.metrics import mean_squared_error , r2_score 

class Evaluation(ABC):
    """
        Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the score for the model
        Args:
            y_true: True labels 
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses mean square error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate MSE")
            mse = mean_squared_error(y_true , y_pred)
            logging.info(f"MSE {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE score : {e}")
            raise e
        
class R2(Evaluation):
    """
        Evaluation strategy that uses R2 score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate R2 score")
            r2 = r2_score(y_true , y_pred)
            logging.info(f"R2 {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 score : {e}")
            raise e
        
class RMSE(Evaluation):
    """
        Evaluation strategy that uses Root mean square error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE : {e}")
            raise e