import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    artifact_dir: str = os.path.join("artifacts")
    train_data_path: str = os.path.join(artifact_dir, "train.csv")
    test_data_path: str = os.path.join(artifact_dir, "test.csv")
    raw_data_path: str = os.path.join(artifact_dir, "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read dataset
            df = pd.read_csv(os.path.join('notebook', 'data', 'stud.csv'))
            logging.info('Successfully read the dataset into a dataframe.')

            # Ensure artifact directory exists
            os.makedirs(self.ingestion_config.artifact_dir, exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Split dataset
            logging.info("Initiating train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Training data saved at: {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved at: {self.ingestion_config.test_data_path}")
            logging.info("Data ingestion completed successfully.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        print(f"Train file created at: {train_path}")
        print(f"Test file created at: {test_path}")
    except CustomException as ce:
        print(f"An error occurred: {ce}")
