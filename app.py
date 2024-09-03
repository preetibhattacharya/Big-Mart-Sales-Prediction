import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import io

# Load Data
st.title("House Price Prediction Dashboard")

def load_data():
    data = pd.read_csv("train_big_mart.csv")
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Model Training"])

if section == "Overview":
    st.header("Data Overview")
    st.write(data.head())
    st.subheader("Number of data points and features")
    st.write(data.shape)
    st.subheader("Dataset Information")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.subheader("Categorical Features")
    st.write(data.select_dtypes(include=['object']).columns.tolist())

elif section == "Data Exploration":



    # Data Distribution Plots
    st.subheader("Data Distribution")
    st.subheader('Numerical Data Distribution')
    column = st.selectbox("Select a column to plot distribution", data.select_dtypes(include=[np.number]).columns)
    color = st.color_picker("Pick a color", '#00f900')
    fig=plt.figure(figsize=(6, 6))
    sns.distplot(data[column], color=color)
    st.pyplot(fig)

    # Categorical Features Plots
    st.subheader('categorical Data Distribution')
    column = st.selectbox("Select a categorical column to plot", data.select_dtypes(include=['object']).columns)
    palette = st.color_picker("Pick a palette color", '#00f900')
    fig1=plt.figure(figsize=(6, 6))
    sns.countplot(x=column, data=data, color=palette)
    st.pyplot(fig1)

    # Correlation Heatmap
    data_encoded = data.copy()
    encoder = LabelEncoder()
    for column in data_encoded.select_dtypes(include=['object']).columns:
        data_encoded[column] = encoder.fit_transform(data_encoded[column])
    st.subheader("Correlation Heatmap")
    fig2=plt.figure(figsize=(17, 8))
    correlation=data_encoded.corr()
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Purples')
    st.pyplot(fig2)

elif section == "Model Training":
    st.header("Model Training and Evaluation")
    
    # Data Preprocessing
    data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

    encoder = LabelEncoder()
    for column in ['Item_Identifier', 'Item_Type', 'Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
        data[column] = encoder.fit_transform(data[column])

    # Splitting Data
    X = data.drop(columns='Item_Outlet_Sales', axis=1)
    Y = data['Item_Outlet_Sales']
    test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=2)

    # Model Training
    st.subheader("Training the Model")
    if st.button("Train Model"):
        Regressor = XGBRegressor()
        Regressor.fit(X_train, Y_train)
        
        # Model Evaluation
        training_data_prediction = Regressor.predict(X_train)
        r2_train = metrics.r2_score(Y_train, training_data_prediction)
        test_data_prediction = Regressor.predict(X_test)
        r2_val = metrics.r2_score(Y_test, test_data_prediction)

        st.subheader("Model Performance")
        st.write(f"Training Data Accuracy: {r2_train}")
        st.write(f"Test Data Accuracy: {r2_val}")


