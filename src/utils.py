import os
import sys
import pickle
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save any Python object (like a model or transformer) to disk using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a pickled Python object from disk.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models: dict):
    """
    Trains and evaluates each model in the provided dictionary using R2 score.

    Returns:
        dict: A dictionary mapping model names to their R2 scores.
    """
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[name] = score
            logging.info(f"{name} R2 Score: {score}")
        return report
    except Exception as e:
        raise CustomException(e, sys)
