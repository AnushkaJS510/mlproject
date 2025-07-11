from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # landing page with a button

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # form page
    else:
        try:
            # Get data from form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race/ethnicity'),
                parental_level_of_education=request.form.get('parental level of education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test preparation course'),
                reading_score=float(request.form.get('reading score')),
                writing_score=float(request.form.get('writing score'))
            )

            # Convert to DataFrame
            final_new_data = data.get_data_as_dataframe()

            # Predict
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(final_new_data)

            if prediction is not None:
                return render_template('home.html', result=round(prediction[0], 2))
            else:
                return render_template('home.html', result="Prediction failed.")

        except Exception as e:
            return render_template('home.html', result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
