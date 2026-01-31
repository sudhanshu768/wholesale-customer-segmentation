import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    layout="centered"
)

st.title(" Wholesale Customer Segmentation")
st.write(
    "This app uses a trained K-Means model to classify wholesale customers "
    "based on their purchasing behavior."
)

@st.cache_resource
def load_model():
    kmeans = joblib.load("kmeans_wholesale.pkl")
    scaler = joblib.load("scaler_wholesale.pkl")
    return kmeans, scaler

kmeans, scaler = load_model()

st.sidebar.header(" Enter Customer Purchase Details")

fresh = st.sidebar.number_input("Fresh", min_value=0.0, value=5000.0)
milk = st.sidebar.number_input("Milk", min_value=0.0, value=3000.0)
grocery = st.sidebar.number_input("Grocery", min_value=0.0, value=4000.0)
frozen = st.sidebar.number_input("Frozen", min_value=0.0, value=2000.0)
detergents = st.sidebar.number_input("Detergents_Paper", min_value=0.0, value=1500.0)
delicassen = st.sidebar.number_input("Delicassen", min_value=0.0, value=1000.0)

user_data = pd.DataFrame([{
    "Fresh": fresh,
    "Milk": milk,
    "Grocery": grocery,
    "Frozen": frozen,
    "Detergents_Paper": detergents,
    "Delicassen": delicassen
}])

st.subheader(" Input Data")
st.dataframe(user_data)

if st.button("Predict Customer Segment"):
    user_scaled = scaler.transform(user_data)
    cluster = kmeans.predict(user_scaled)[0]

    st.success(f" This customer belongs to **Cluster {cluster}**")

    cluster_descriptions = {
        0: "Retail-oriented buyers (high grocery & detergents)",
        1: "Regular small-to-medium buyers",
        2: "High-volume wholesale buyers",
        3: "Specialty / hotel & restaurant buyers"
    }

    st.subheader(" Cluster Interpretation")
    st.write(cluster_descriptions.get(cluster, "No description available"))
