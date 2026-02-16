import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Risk Assessment AI", page_icon="🤖", layout="centered")

# Title & Header
st.title("🤖 Project Risk Assessment AI")
st.markdown("---")
st.write("Enter the project details below to predict the risk level.")

# Load Model & Columns
@st.cache_resource
def load_artifacts():
    # Check current directory then models directory
    model_path = "models/risk_model.pkl"
    cols_path = "models/train_columns.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Did you run 'train_model.py'?")
        return None, None
        
    model = joblib.load(model_path)
    columns = joblib.load(cols_path)
    return model, columns

model, model_columns = load_artifacts()

if model is not None:
    # 📝 Input Form (Better than typing text)
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input("Project Budget (USD)", min_value=160000.0, value=200000.0, step=10000.0)
            timeline = st.number_input("Estimated Timeline (Months)", min_value=2.0, value=12.0, step=1.0)
            team_size = st.number_input("Team Size", min_value=2.0, value=10.0, step=1.0)
            success_rate = st.slider("Previous Success Rate", 0.15, 1.0, 0.7, help="Ratio of successful past projects")

        with col2:
            complexity = st.slider("Complexity Score", 1, 10, 5, help="1=Easy, 10=Super Hard")
            # Schedule Pressure in data is 0 to 0.58. Mapping 1-10 to 0.05-0.6
            pressure_input = st.slider("Schedule Pressure", 1, 10, 5, help="1=Relaxed, 10=Impossible Deadline")
            # Resources in data is 0.3 to 1.0. Mapping 1-10 to 0.1-1.0
            resources_input = st.slider("Resource Availability", 1, 10, 7, help="1=Scarce, 10=Abundant")
            req_stability = st.selectbox("Requirement Stability", ["Low", "Medium", "High"], index=1)
            
        # Optional defaults
        vendor_reliability = 0.7 # Default (0-1 scale)
        past_projects = 5 # Default
        budget_utilization = 0.8 # Default (0.6-1.3 scale)
        
        submitted = st.form_submit_button("🔮 Predict Risk")

    if submitted:
        # 🟢 SCALING FIX: Map 1-10 inputs to Data Ranges
        pressure = pressure_input / 15.0  # Maps 10 -> ~0.66 (Close to max 0.58)
        resources = resources_input / 10.0 # Maps 10 -> 1.0
        
        # 1. Create Dataframe from inputs
        input_data = pd.DataFrame({
            'Project_Budget_USD': [budget],
            'Estimated_Timeline_Months': [timeline],
            'Team_Size': [team_size],
            'Complexity_Score': [complexity],
            'Schedule_Pressure': [pressure],
            'Requirement_Stability': [req_stability], 
            'Past_Similar_Projects': [past_projects],
            'Previous_Delivery_Success_Rate': [success_rate],
            'Vendor_Reliability_Score': [vendor_reliability],
            'Resource_Availability': [resources],
            'Budget_Utilization_Rate': [budget_utilization]
        })

        # 2. Feature Engineering (EXACTLY SAME AS TRAINING)
        # Map Requirement Stability
        input_data['Requirement_Stability'] = input_data['Requirement_Stability'].map({
            'Low': 1, 'Medium': 2, 'High': 3
        })

        # Derived Features
        input_data['Budget_per_Month'] = input_data['Project_Budget_USD'] / input_data['Estimated_Timeline_Months']
        input_data['Workload_Index'] = input_data['Complexity_Score'] / input_data['Team_Size']

        input_data['Experience_Buffer'] = (
            input_data['Past_Similar_Projects'] *
            input_data['Previous_Delivery_Success_Rate']
        )

        input_data['Org_Strength_Index'] = (
            input_data['Vendor_Reliability_Score'] +
            input_data['Resource_Availability'] +
            input_data['Previous_Delivery_Success_Rate']
        ) / 3

        input_data['Pressure_x_Complexity'] = (
            input_data['Schedule_Pressure'] *
            input_data['Complexity_Score']
        )
        input_data['Risk_Intensity'] = (
            input_data['Schedule_Pressure'] *
            input_data['Budget_Utilization_Rate']
        )

        input_data['Team_Pressure'] = (
            input_data['Workload_Index'] *
            input_data['Schedule_Pressure']
        )

        input_data['Log_Budget'] = np.log1p(input_data['Project_Budget_USD'])

        # 3. One Hot Encoding (This might create fewer columns than training)
        input_df_encoded = pd.get_dummies(input_data, drop_first=True)

        # 4. Reindex (The Magic Fix 🪄)
        # This forces the input data to have the EXACT columns as training data.
        # It fills missing columns with 0.
        final_input = input_df_encoded.reindex(columns=model_columns, fill_value=0)

        # 5. Prediction
        prediction = model.predict(final_input)[0]
        
        try:
            proba = model.predict_proba(final_input)[0][1]
        except:
            proba = 0.5 # Fallback if model logic varies

        # 6. Display Result
        st.markdown("---")
        
        # Risk Threshold (Lowered to 0.35 to be more sensitive/conservative)
        THRESHOLD = 0.35
        
        # Override: If Pressure is Extreme (>0.5) OR Resources Very Low (<0.2), Force High Risk
        critical_risk_factors = (pressure > 0.5) or (resources < 0.2)
        
        if (proba > THRESHOLD) or critical_risk_factors:
            st.error(f"⚠️ **High Risk Project**")
            st.metric("Risk Probability", f"{proba:.2%}")
            
            if critical_risk_factors:
                 st.write("🔴 **Critical Driver:** Extreme Pressure or Resource Shortage detected!")
            
            st.warning("Recommendation: Increase budget buffer or extend timeline immediately.")
        else:
            st.success(f"✅ **Low Risk Project**")
            st.metric("Risk Probability", f"{proba:.2%}")
            st.info("The project parameters look stable.")
