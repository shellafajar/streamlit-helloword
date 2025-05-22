import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = sns.load_dataset('iris')

st.title("📊 Dashboard Dataset Iris")
st.write("Tampilan awal data dan statistik deskriptif.")

# Tampilkan data
st.subheader("Data Iris")
st.dataframe(df)

# Statistik deskriptif
st.subheader("Statistik Deskriptif")
st.write(df.describe())

# Plot
st.subheader("Visualisasi Sepal Width vs Sepal Length")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=ax)
st.pyplot(fig)
