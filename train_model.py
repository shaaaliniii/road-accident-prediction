import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load datasets
acc = pd.read_csv("final-yr-projectqA/Accidents.csv")
veh = pd.read_csv("final-yr-projectqA/Vehicles.csv")
cas = pd.read_csv("final-yr-projectqA/Casualties.csv")

print("✅ Datasets loaded")

# Merge datasets
df = acc.merge(veh, on="Accident_Index", how="left")
df = df.merge(cas, on="Accident_Index", how="left")

print("✅ Datasets merged")

# Target column
target_col = "Accident_Severity"

# Drop unnecessary columns
drop_cols = ["Accident_Index", "LSOA_of_Accident_Location"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Drop missing values
df = df.dropna()

# Drop date/time
df = df.drop(columns=["Date", "Time"], errors="ignore")

# ⚡ IMPORTANT: reduce size (otherwise slow)
df = df.sample(min(50000, len(df)))

# Convert categorical → numeric
df = pd.get_dummies(df)

# Split
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
print("🎯 Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "litemodel.sav")
joblib.dump(X.columns.tolist(), "columns.pkl") 

print("✅ Model trained & saved!")