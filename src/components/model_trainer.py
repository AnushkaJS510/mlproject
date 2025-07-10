import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model  # Both functions must exist in utils.py


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.array, test_array: np.array) -> float:
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "K-Neighbors Regressor": KNeighborsRegressor(),
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model evaluation report: {model_report}")

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 score >= 0.6", sys)

            logging.info(f"Best model selected: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            logging.error("Error occurred during model training", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    # 1️⃣ Ingest raw data
    from src.components.data_ingestion import DataIngestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    # 2️⃣ Transform the ingested data
    from src.components.data_transformation import DataTransformation
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)
    
    # 3️⃣ Train & evaluate models
    trainer = ModelTrainer()
    best_r2 = trainer.initiate_model_trainer(train_arr, test_arr)
    
    print(f"\n✅ End-to-end pipeline finished. Best R² = {best_r2:.4f}")
