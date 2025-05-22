import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Konfigurasi halaman
st.set_page_config(page_title="Prediction")
st.title("Predictions")
st.header("Predictions")

# Load Dataset
df = pd.read_csv("model/iris.csv")  # Jika tersedia lokal
dataset = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f91306ad21198e34c1b4f7fdf79f59d2b1dcb9e/iris.csv")

# Tampilkan DataFrame
st.subheader("📊 Iris Dataset")
st.write(dataset)

# Pisahkan fitur dan target
X = dataset.drop("variety", axis=1)
y = dataset["variety"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load atau latih model
try:
    model = joblib.load("model/iris_model.pkl")
except:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "model/iris_model.pkl")

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("🎯 Accuracy")
st.write(f"{acc:.2f}")

st.subheader("📄 Classification Report")
st.text(classification_report(y_test, y_pred))

# Input untuk prediksi baru
st.subheader("🔍 Predict New Sample")
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    new_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=["sepal.length", "sepal.width", "petal.length", "petal.width"])
    prediction = model.predict(new_data)
    st.success(f"Predicted variety: {prediction[0]}")

