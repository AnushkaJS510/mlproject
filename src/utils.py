import os
import sys
import dill
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves the given object to a file using dill serialization.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
