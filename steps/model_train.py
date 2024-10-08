# this is the 3nd step
import logging
import pandas as pd 
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
        Trains the model on the ingest data
        Args:
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_test: pd.DataFrame,
    """
    model = None
   
    try:
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model       
        else:
            raise ValueError(f"Model {config.model_name} not supported")

    except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
    
