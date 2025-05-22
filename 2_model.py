import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = sns.load_dataset('iris')

st.title("🤖 Evaluasi Model Machine Learning")

# Pisahkan fitur dan target
X = df.drop("species", axis=1)
y = df["species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
acc = accuracy_score(y_test, y_pred)
st.subheader("Akurasi Model")
st.write(f"{acc:.2%}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
