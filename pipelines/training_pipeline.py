from zenml import pipeline
import pandas as pd

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df 
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    """
    ingest the data , cleans the data , train and evaluate the model
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test , y_train, y_test)
    mse , r2 , rmse = evaluate_model(model, X_test, y_test)
