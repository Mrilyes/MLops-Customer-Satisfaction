# this is the 2rd step
import logging
import pandas as pd 
from zenml import step
from src.data_cleaning import DataPreprocessStategy, DataDivideStrategy, DataCleaning

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    
    """
     Cleans the data and divides it into train and test

     Args:
        df: Raw data

     Returns:
        X_train , X_test , y_train and y_test
    
    """
    try:
        preprocess_strategy = DataPreprocessStategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)

        X_train, X_test , y_train , y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
        return X_train, X_test , y_train , y_test
    
    except Exception as e:
        logging.error(f"Error in cleaning data {e}")
        raise e