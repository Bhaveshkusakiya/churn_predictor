import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('churn_model.pkl', 'rb'))

# Streamlit App
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📈 Customer Churn Prediction App")
st.write("Upload a customer CSV file and predict if they are likely to churn.")

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data:")
    st.dataframe(data)

    # Preprocessing
    # Drop customerID if exists
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)

    # Ensure same data format
    try:
        predictions = model.predict(data)

        data['Churn_Prediction'] = predictions

        st.subheader("Prediction Results:")
        st.dataframe(data)

        st.subheader("Churn Prediction Summary:")
        st.bar_chart(data['Churn_Prediction'].value_counts())

    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.info("Make sure your CSV columns match the training data format!")

else:
    st.info("Please upload a valid CSV file to start prediction.")

# Footer
st.markdown("""
    <hr style="margin-top: 2rem; margin-bottom: 1rem;">
    <div style="text-align: center;">Made with ❤️ by Bhavesh Kusakiya</div>
""", unsafe_allow_html=True)
