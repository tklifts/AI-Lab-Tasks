from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and encoders once at startup
model = joblib.load("model.pkl")
le_home = joblib.load("le_home.pkl")
le_away = joblib.load("le_away.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    home_xg = float(request.form['homeXG'])
    away_xg = float(request.form['awayXG'])
    home_team = request.form['homeTeam']
    away_team = request.form['awayTeam']

    try:
        home_encoded = le_home.transform([home_team])[0]
        away_encoded = le_away.transform([away_team])[0]
    except ValueError:
        return "Error: One of the team names is not recognized."

    features = [[home_xg, away_xg, home_encoded, away_encoded]]
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction=f"Predicted Result: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
