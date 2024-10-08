# this is the 1st step
# this file will consist the steps where we will ingest the data in it

import logging 
import pandas as pd 
from zenml import step

class IngestData:
    """
        Args:
            data_path: path of the data 
    """
    def __init__(self,data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path

    Args:
        data_path: path of the data

    Returns:
        pd.DataFrame : the ingested data        
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e 