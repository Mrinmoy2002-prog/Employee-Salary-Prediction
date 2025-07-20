import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
# This must be the first Streamlit command in your script.
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="wide"
)

# --- IMPORTANT NOTE ---
# This app now uses a single pipeline file that contains both the preprocessor
# and the model. This is the standard and most reliable way to deploy models.
#
# In your training notebook, you should create and save the pipeline like this:
# from sklearn.pipeline import Pipeline
#
# full_pipeline = Pipeline(steps=[
#     ('preprocessor', ct),
#     ('classifier', your_model_object)
# ])
# full_pipeline.fit(X_train, y_train)
# joblib.dump(full_pipeline, 'full_pipeline.pkl')

# --- Load The Complete Pipeline ---
try:
    pipeline = joblib.load("full_pipeline.pkl")
except FileNotFoundError:
    st.error("Error: The file 'full_pipeline.pkl' was not found. Please ensure you have trained and saved the complete pipeline from your notebook.")
    st.stop()


# --- UI Setup ---
st.title("💰 Income Prediction App")
st.markdown("Predict whether an individual's income is likely to be more than $50K or less than or equal to $50K per year based on their demographic and employment data.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Features")
st.sidebar.markdown("Please provide the details of the individual:")

# --- Input Fields (Updated based on your preprocessing) ---

# Numerical Inputs
age = st.sidebar.slider("Age", 17, 75, 35)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 12285, 1490400, 178356)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# Categorical Inputs with predefined options
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov']
workclass = st.sidebar.selectbox("Work Class", workclass_options)

education_options = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate']
education = st.sidebar.selectbox("Education Level", education_options)

education_map = {
    '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9,
    'Prof-school': 10, 'Assoc-acdm': 11, 'Assoc-voc': 12, 'Some-college': 13,
    'Bachelors': 14, 'Masters': 15, 'Doctorate': 16
}
educational_num = education_map.get(education, 9)

marital_status_options = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)

occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
occupation = st.sidebar.selectbox("Occupation", occupation_options)

relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
relationship = st.sidebar.selectbox("Relationship", relationship_options)

race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
race = st.sidebar.selectbox("Race", race_options)

gender = st.sidebar.radio("Gender", ['Male', 'Female'])

native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands']
native_country = st.sidebar.selectbox("Native Country", native_country_options)


# --- Prediction Logic ---
# Create a DataFrame from the inputs in the correct order.
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data for Model")
st.dataframe(input_data)

# Prediction button
if st.button("Predict Income", key='predict_button'):
    try:
        # The pipeline handles both preprocessing and prediction in one step
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0]

        st.markdown("---")
        st.write("### 🔮 Prediction Result")
        
        if prediction == '<=50K':
            st.success(f"**Predicted Income:** {prediction}")
        else:
            st.warning(f"**Predicted Income:** {prediction}")

        st.write("#### Confidence Score")
        st.write(f"**Probability of earning >50K:** {prediction_proba[1]:.2f}")
        st.write(f"**Probability of earning ≤50K:** {prediction_proba[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Batch Prediction ---
st.markdown("---")
st.markdown("### 📂 Batch Prediction")
st.markdown("Upload a CSV file with the required columns to get predictions for multiple individuals.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        
        # Create a copy to avoid modifying the user's original view
        data_to_predict = batch_data.copy()

        # Ensure educational-num is mapped correctly for batch data
        if 'education' in data_to_predict.columns:
            data_to_predict['educational-num'] = data_to_predict['education'].map(education_map)
            data_to_predict.drop(columns=['education'], inplace=True)
        
        st.write("Uploaded data preview:", batch_data.head())
        
        # Make predictions using the complete pipeline
        batch_preds = pipeline.predict(data_to_predict)
        batch_data['Predicted_Income'] = batch_preds
        
        st.write("✅ **Predictions Complete!**")
        st.dataframe(batch_data.head())
        
        # Provide download link
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions CSV",
            csv,
            file_name='predicted_income_data.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"An error occurred during batch prediction: {e}")
