from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the pickle file
try:
    with open('Wine_quality_model2.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    model = None



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and model:
        try:
            # Collect all feature inputs
            input_features = [float(request.form[f'feature{i}']) for i in range(1, 12)]
            features = np.array(input_features).reshape(1, -1)

            prediction = model.predict(features)

            # Determine the quality based on prediction
            quality = 'good' if np.argmax(prediction) == 1 else 'bad'

            return render_template('index.html', prediction_text=f'Wine quality is {quality}')

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

    return render_template('index.html', prediction_text='')

if __name__ == "__main__":
    app.run(debug=True)