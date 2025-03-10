from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('random_forest_model.pkl')

# Load the dataset to display car names and details
data = pd.read_csv('auto.csv')

# Define the features for the input form
FEATURES = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']

# Prepare top 20 cars with low emissions to display on the home page
data['predicted_emission'] = model.predict(data[FEATURES])
top_20_cars = data.sort_values(by='predicted_emission').head(20)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the form
        user_input = [
            float(request.form['mpg']),
            int(request.form['cylinders']),
            float(request.form['displacement']),
            float(request.form['horsepower']),
            float(request.form['weight']),
            float(request.form['acceleration']),
            int(request.form['year']),
            int(request.form['origin'])
        ]
        
        # Convert the input to a DataFrame for prediction
        user_input_df = pd.DataFrame([user_input], columns=FEATURES)
        
        # Predict emission for user input
        predicted_emission = model.predict(user_input_df)[0]
        
        # Find cars within a similar emission range
        emission_tolerance = 50  # Adjust tolerance as needed
        data['emission_diff'] = abs(data['predicted_emission'] - predicted_emission)
        similar_cars = data.sort_values(by='emission_diff').head(20)

        return render_template('index.html', 
                               top_20_cars=top_20_cars.to_dict(orient="records"), 
                               user_input=user_input, 
                               predicted_emission=predicted_emission, 
                               similar_cars=similar_cars.to_dict(orient="records"))
    return render_template('index.html', top_20_cars=top_20_cars.to_dict(orient="records"))

if __name__ == '__main__':
    app.run(debug=True)
