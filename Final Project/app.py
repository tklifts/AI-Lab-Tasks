from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the dataset
file_path = (r"FastFoodNutritionMenuV2 (1).csv")
df = pd.read_csv(file_path)
df.columns = df.columns.str.replace('\n', ' ', regex=True).str.strip()
for col in df.columns:
    if col not in ['Company', 'Item']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df_cleaned = df.dropna()

# Prepare features and target
X = df_cleaned.drop(columns=['Company', 'Item', 'Calories'])
y = df_cleaned['Calories']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Define routes
@app.route('/')
def home():
    # Pass feature column names to the template
    return render_template('index.html', columns=X.columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = [float(request.form[col]) for col in X.columns]
    input_array = np.array(input_data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    return jsonify({'prediction': f'Predicted Calories: {prediction:.2f}'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)