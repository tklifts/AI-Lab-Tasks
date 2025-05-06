import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
file_path = (r"FastFoodNutritionMenuV2 (1).csv")  # Adjust path if needed
df = pd.read_csv(file_path)

# Step 2: Clean column names
df.columns = df.columns.str.replace('\n', ' ', regex=True).str.strip()

# Step 3: Convert columns to numeric (excluding 'Company' and 'Item')
for col in df.columns:
    if col not in ['Company', 'Item']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 4: Drop rows with missing values
df_cleaned = df.dropna()

# Step 5: Prepare features and target
X = df_cleaned.drop(columns=['Company', 'Item', 'Calories'])
y = df_cleaned['Calories']

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 9: Print performance
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.5f}")
