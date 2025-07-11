import pandas as pd
import os
import pickle

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        return pd.DataFrame({
            'gender': [self.gender],
            'race/ethnicity': [self.race_ethnicity],  # ✅ matches model
            'parental level of education': [self.parental_level_of_education],
            'lunch': [self.lunch],
            'test preparation course': [self.test_preparation_course],
            'reading score': [self.reading_score],
            'writing score': [self.writing_score]
        })


class PredictPipeline:
    def __init__(self):
        # Load model and preprocessor paths
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

    def predict(self, input_df):
        try:
            processed_input = self.preprocessor.transform(input_df)
            prediction = self.model.predict(processed_input)
            return prediction
        except Exception as e:
            print("Prediction error:", e)
            return None
