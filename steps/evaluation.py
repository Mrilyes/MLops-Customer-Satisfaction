# this is the 4th step
import logging

import pandas as pd 
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from sklearn.base import RegressorMixin
from src.evaluation import MSE , R2 , RMSE

@step
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    ) -> Tuple[
        Annotated[float,"mse"],
        Annotated[float,"r2_score"],
        Annotated[float,"rmse"],
    ]:
    
    """
    Evaluates the model on the ingested data.
    Args:
        df: the ingested data
    """

    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        return mse , r2 , rmse
    
    except Exception as e:
        logging.error(f"Error in evaluation model {e}")
        raise e 
    
# compare this sinppet from steps/model_train.py