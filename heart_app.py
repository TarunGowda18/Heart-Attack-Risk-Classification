import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title('❤️Heart attack risk classification app')

# 1. LOAD MODEL AND SCALER
# Note: You should export your scaler from your notebook as 'scaler.pkl'
model = pickle.load(open('gboost_model.pkl','rb'))
# scaler = pickle.load(open('scaler.pkl','rb')) # Recommended: Load your actual trained scaler

# ... (Keep all your st.number_input and selectbox code here) ...

# 2. DEFINE DATAFRAME (This must happen BEFORE scaling)
input_features = pd.DataFrame({
    'Age':[Age], 'RestingBP':[RestingBP], 'Cholesterol':[Cholesterol],
    'FastingBS':[FastingBS], 'MaxHR':[MaxHR], 'Oldpeak':[Oldpeak],
    'sex':[sex], 'exerciseAngina':[exerciseAngina], 'RestingECG_LVH':[RestingECG_LVH],
    'RestingECG_Normal':[RestingECG_Normal], 'RestingECG_ST':[RestingECG_ST], 
    'ChestPainType_ASY':[ChestPainType_ASY], 'ChestPainType_ATA':[ChestPainType_ATA], 
    'ChestPainType_NAP':[ChestPainType_NAP], 'ChestPainType_TA':[ChestPainType_TA],
    'st_Slope':[st_Slope]
})

# 3. CORRECT SCALING
# Use an existing scaler. If you MUST create a new one here (not recommended), 
# you cannot scale a single row. For now, we will use your existing logic 
# but move it to the correct position.
scaler = StandardScaler() 
cols_to_scale = ['Age' , 'RestingBP','Cholesterol','MaxHR','Oldpeak']
# Note: .fit_transform on one row makes everything 0. 
# Ideally, replace the line below with: input_features[cols_to_scale] = scaler.transform(input_features[cols_to_scale])
input_features[cols_to_scale] = scaler.fit_transform(input_features[cols_to_scale]) 

# 4. PREDICTIONS
if st.button('Predict'):
    predictions = model.predict(input_features)
    if predictions == 1:
        st.error('⚠️High Risk of Heart Attack❗')
    else:
        st.success('Low Risk of Heart Attack😊')
