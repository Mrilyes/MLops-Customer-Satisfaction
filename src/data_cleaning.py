import logging
from abc import ABC,abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


# The pattern used in this file called "Stategy pattern"

class DataStrategy(ABC):
    """
        Abstract class defining stategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) ->pd.DataFrame: # type: ignore
        pass


class DataPreprocessStategy(DataStrategy):
    """
        Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Data preprocessing strategy which preprocesses the data.
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        
        except Exception as e:
            logging.error(f"Error in processing data{e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Dividing / splitting data into train and test
        """
        try:
            X = data.drop(['review_score'], axis=1)
            y = data['review_score']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Ensure y_train and y_test are DataFrames
            y_train = pd.DataFrame(y_train).reset_index(drop=True)
            y_test = pd.DataFrame(y_test).reset_index(drop=True)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e



class DataCleaning:
    """
        Class for cleaning data which process the data and divide it into train and test
    """

    def __init__(self, data:pd.DataFrame, strategy:DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> pd.DataFrame:
        """
            Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data {e}")


# example of the purpose of strategy

# if __name__ == "__main__":
#     data = pd.read_csv("/home/lenovo/Desktop/MyProjects/MLops/my work : MLops-Brazilian E-Commerce by Olist/data/olist_customers_dataset.csv")
#     data_cleaning = DataCleaning(data , DataPreprocessStategy())
#     data_cleaning.handle_data()