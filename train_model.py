import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and clean data
df = pd.read_csv("your_file.csv")  # Replace with your actual file path
df_clean = df.dropna(subset=['Result', 'homeXG', 'awayXG'])

# Encode team names
le_home = LabelEncoder()
le_away = LabelEncoder()
df_clean['Home Team Enc'] = le_home.fit_transform(df_clean['Home Team'])
df_clean['Away Team Enc'] = le_away.fit_transform(df_clean['Away Team'])

# Train model
X = df_clean[['homeXG', 'awayXG', 'Home Team Enc', 'Away Team Enc']]
y = df_clean['Result']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(le_home, "le_home.pkl")
joblib.dump(le_away, "le_away.pkl")
