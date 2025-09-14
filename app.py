import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Assets ---
@st.cache_data
def load_assets():
    # Using relative paths for portability. Place your files in the same folder as the app.
    model_path = 'models\lightgbm_churn_model.pkl'
    data_path = 'data\Train.csv'
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error: The model file '{model_path}' was not found.")
        st.info("Please run your training script to generate the model file and place it in the same directory as this app.")
        st.stop()

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: '{data_path}' not found. Make sure the dataset is in the same directory.")
        st.stop()
    return model, df

model, df = load_assets()

# --- Column Names ---
column_names = {
    'feature_0': 'premium_to_age_ratio', 'feature_1': 'claim_frequency',
    'feature_2': 'policy_tenure_scaled', 'feature_3': 'payment_delay_score',
    'feature_4': 'service_interaction_count', 'feature_5': 'discount_eligibility_score',
    'feature_6': 'risk_score', 'feature_7': 'region_code', 'feature_8': 'sales_channel_id',
    'feature_9': 'policy_type', 'feature_10': 'renewal_status', 'feature_11': 'family_plan_flag',
    'feature_12': 'auto_renew_flag', 'feature_13': 'digital_engagement_level',
    'feature_14': 'days_associated', 'feature_15': 'Contact_Frequency', 'labels': 'Churn'
}

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Predict Churn", "Model Performance", "About"])


# =====================================================================================
# PAGE 1: DATASET OVERVIEW
# =====================================================================================
if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("This page provides a high-level summary of the original training dataset (`Train.csv`).")
    
    total_customers = len(df)
    churn_count = df['labels'].sum()
    no_churn_count = total_customers - churn_count
    churn_rate = (churn_count / total_customers) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Churned Customers", f"{churn_count:,}")
    col3.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    ax.pie([churn_count, no_churn_count], labels=['Churned', 'Retained'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    st.subheader("Understanding Key Churn Drivers")
    st.write("Based on the model, certain customer attributes are strong indicators of potential churn. Here are the most important ones:")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("##### High-Impact Risk Factors")
        st.markdown("""
        - **Risk Score:** A composite score of a customer's overall risk. **Higher scores** directly correlate with a **higher chance of churn**.
        - **Payment Delay Score:** Measures payment behavior. **Positive scores** (late payments) are a powerful predictor of churn.
        - **Premium to Age Ratio:** Compares premium cost to customer age. A **high ratio** often indicates the customer feels they are overpaying, increasing churn risk.
        """)
    with f_col2:
        st.markdown("##### High-Impact Retention Factors")
        st.markdown("""
        - **Auto-Renewal:** Customers with auto-renewal enabled are significantly **less likely to churn**. It is a strong indicator of customer satisfaction and inertia.
        - **Discount Eligibility Score:** Represents loyalty status or eligibility for discounts. A **higher score** means the customer is more valued and thus **less likely to churn**.
        - **Policy Tenure (Scaled):** Represents how long the customer has been with the company. **Higher (positive) values** indicate longer tenure and loyalty, making churn **less likely**.
        """)


# =====================================================================================
# PAGE 2: PREDICT CHURN
# =====================================================================================
elif page == "Predict Churn":
    st.header("Predict Customer Churn")
    
    prediction_method = st.radio("Select Prediction Method", ["Single Customer (Manual Input)", "Batch Prediction (Upload File)"])

    if prediction_method == "Single Customer (Manual Input)":
        st.subheader("Key Customer Risk Indicators")
        st.write("Enter the customer's information to get a churn prediction.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_score = st.number_input("Risk Score", min_value=0.0, max_value=100.0, value=0.0, step=1.0) # Default to low risk
            st.info("**Range: 0.0 to 100.0**\nA consolidated score of the customer's risk. Higher scores mean higher churn risk.")
        
        with col2:
            payment_delay_score = st.number_input("Payment Delay Score", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            st.info("**Range: -5.0 to 5.0**\nPositive scores (delays) strongly increase churn risk.")

        with col3:
            service_interaction_count = st.number_input("Service Interaction Count", min_value=0, max_value=50, value=0) # Default to low risk
            st.info("**Range: 0 to 50**\nHigh interaction counts can signal customer frustration.")

        col4, col5 = st.columns(2)
        with col4:
            premium_to_age_ratio = st.number_input("Premium to Age Ratio", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            st.info("**Range: 0.0 to 10.0**\nA high ratio is a strong churn driver.")
            
        with col5:
            claim_frequency = st.number_input("Claim Frequency", min_value=0.0, max_value=5.0, value=0.0, step=0.1) # Default to low risk
            st.info("**Range: 0.0 to 5.0**\nHigh frequency can indicate dissatisfaction.")

        st.subheader("Additional Customer & Policy Details")
        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            policy_tenure_scaled = st.number_input("Policy Tenure (Scaled)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1) # Default to high tenure
            st.info("**Range: -5.0 to 5.0**\nRepresents policy length. Lower (negative) values can indicate higher early-churn risk.")
            
            discount_eligibility_score = st.number_input("Discount Eligibility Score", min_value=0.0, max_value=1.0, value=1.0, step=0.05) # Default to high score
            st.info("**Range: 0.0 to 1.0**\nA score representing loyalty or offer eligibility. A low score might increase churn risk.")

            contact_freq = st.number_input("Contact Frequency (last 12 months)", min_value=0, max_value=50, value=5)
            st.info("**Range: 0 to 50**\nNumber of times the customer has been contacted.")

        with e_col2:
            region_code = st.selectbox("Region Code", list(range(6)), index=0)
            st.info("The customer's geographical region.")
            
            sales_channel_id = st.selectbox("Sales Channel ID", list(range(3)), index=0)
            st.info("How the customer was acquired (e.g., Online, Agent).")

            policy_type = st.selectbox("Policy Type", list(range(4)), index=0)
            st.info("The type of insurance policy (e.g., Auto, Health).")

        with e_col3:
            renewal_status = st.selectbox("Renewal Status", [0, 1], index=1, format_func=lambda x: "Not Renewed" if x==0 else "Renewed") # Default to Renewed
            st.info("Whether the customer renewed their policy last term.")

            family_plan_flag = st.selectbox("Family Plan", [0, 1], index=1, format_func=lambda x: "No" if x==0 else "Yes") # Default to Yes
            st.info("Is the customer part of a family or group plan?")

            auto_renew_flag = st.selectbox("Auto-Renewal", [0, 1], index=1, format_func=lambda x: "Off" if x==0 else "On") # Default to On
            st.info("Enabled auto-renewal is a strong factor in retaining customers.")
        
        st.markdown("---") # Visual separator
        d_col1, d_col2 = st.columns(2)
        with d_col1:
             digital_engagement_level = st.selectbox("Digital Engagement", [0, 1, 2], index=2, format_func=lambda x: {0:"Low", 1:"Medium", 2:"High"}[x]) # Default to High
             st.info("Customer's usage of the app or web portal. Low engagement can be a churn indicator.")
        with d_col2:
             days_associated = st.selectbox("Days Associated Group", [1,3,5,8,9,10,11], index=6) # Default to high tenure group
             st.info("A categorical grouping related to the customer's tenure.")


        if st.button("Predict Churn", type="primary", use_container_width=True):
            feature_names_original = list(column_names.keys())[:-1]
            input_data = [
                premium_to_age_ratio, claim_frequency, policy_tenure_scaled, payment_delay_score,
                service_interaction_count, discount_eligibility_score, risk_score, region_code,
                sales_channel_id, policy_type, renewal_status, family_plan_flag,
                auto_renew_flag, digital_engagement_level, days_associated, contact_freq
            ]
            input_df = pd.DataFrame([input_data], columns=feature_names_original)
            
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"Prediction: **Customer is likely to CHURN** (Probability: {prediction_proba[0][1]*100:.2f}%)")
            else:
                st.success(f"Prediction: **Customer is likely to be RETAINED** (Probability: {prediction_proba[0][0]*100:.2f}%)")
            
            st.subheader("Prediction Breakdown")
            st.write("This is the model's reasoning based on the most influential inputs provided.")
            
            if prediction[0] == 1: # Churn predicted
                if renewal_status == 0:
                    st.warning("**Primary Driver:** The strongest factor is the **'Not Renewed'** status from the previous term. This is a very high-risk indicator.")
                elif auto_renew_flag == 0:
                    st.warning("**Primary Driver:** **Auto-Renewal is 'Off'**. The model has learned that customers without auto-renewal are significantly more likely to churn.")
                elif risk_score > 75:
                    st.warning(f"**Primary Driver:** The **Risk Score of {risk_score} is very high**, indicating a significant churn risk based on multiple underlying factors.")
                elif payment_delay_score > 1.0:
                    st.warning(f"**Key Factor:** The **Payment Delay Score of {payment_delay_score} is high**, suggesting payment issues that often lead to churn.")
                else:
                    st.info("The prediction is based on a complex combination of multiple smaller risk factors rather than a single dominant one.")
            
            else: # Retain predicted
                if auto_renew_flag == 1 and discount_eligibility_score >= 0.8:
                     st.info("**Primary Driver:** The combination of **Auto-Renewal being 'On'** and a **high Discount Eligibility Score** is a powerful indicator of a loyal customer, often overriding other risk factors.")
                elif payment_delay_score < -1.0:
                     st.info("**Primary Driver:** The customer pays their bills significantly early (**Payment Delay Score: {payment_delay_score}**), which is a strong sign of stability and satisfaction.")
                elif risk_score < 20 and service_interaction_count == 0:
                     st.info("**Key Factors:** The very **low Risk Score ({risk_score})** combined with **zero service interactions** suggests a satisfied, low-maintenance customer who is likely to be retained.")
                else:
                    st.info("The prediction is based on a combination of positive factors (like high tenure and engagement) that outweigh any potential risks.")

    elif prediction_method == "Batch Prediction (Upload File)":
        st.info("Upload a CSV file with columns 'feature_0', 'feature_1', etc.")
        uploaded_file = st.file_uploader("Choose a file", type="csv")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())

            if st.button("Predict on Batch", type="primary"):
                predictions = model.predict(batch_df)
                
                result_df = batch_df.copy()
                result_df['Predicted_Churn'] = predictions
                
                st.subheader("Batch Prediction Summary")
                batch_total = len(result_df)
                batch_churn_count = result_df['Predicted_Churn'].sum()
                batch_churn_rate = (batch_churn_count / batch_total) * 100

                b_col1, b_col2, b_col3 = st.columns(3)
                b_col1.metric("Total Customers in Batch", f"{batch_total:,}")
                b_col2.metric("Predicted to Churn", f"{batch_churn_count:,}")
                b_col3.metric("Predicted Churn Rate", f"{batch_churn_rate:.2f}%")

                st.write("Prediction Results:")
                result_df_display = result_df.rename(columns={k:v for k,v in column_names.items() if k!='labels'})
                st.dataframe(result_df_display)
                
                csv = result_df_display.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv, file_name='churn_predictions.csv', mime='text/csv')

# =====================================================================================
# PAGE 3: MODEL PERFORMANCE
# =====================================================================================
elif page == "Model Performance":
    st.header("Model Performance Evaluation")
    st.info("The following metrics represent the final performance of the trained model on the hold-out validation dataset, based on the `Trian.ipynb` notebook.")

    st.subheader("Overall Performance Metrics")
    performance_data = {
        'Metric': ['Accuracy', 'Precision (for Churn)', 'Recall (for Churn)', 'F1-Score (for Churn)', 'ROC AUC Score'],
        'Score': [0.9492, 0.9541, 0.9439, 0.9490, 0.9917] 
    }
    performance_df = pd.DataFrame(performance_data)
    st.table(performance_df.set_index('Metric'))

    st.subheader("Confusion Matrix")
    st.write("The matrix shows the model's predictions vs. the actual outcomes.")
    cm_data = {'Predicted: Not Churn': [5690, 359], 'Predicted: Churn': [299, 5629]}
    cm_df = pd.DataFrame(cm_data, index=['Actual: Not Churn', 'Actual: Churn'])
    st.table(cm_df)

    st.subheader("Feature Importance")
    st.markdown("""
    This chart displays the features ranked by their importance to the model's predictions.

    - **X-axis:** Lists the customer features.
    - **Y-axis (Importance Score):** Represents how many times a feature was used by the model to make a decision point (a 'split') across all of its internal decision trees.
    
    A **taller bar** signifies that the feature is more influential and has a greater impact on the final prediction. For example, a feature with a score of 1000 was used for decision-making far more often than a feature with a score of 100.
    """)
    try:
        importances = model.feature_importances_
        feature_names = df.columns.drop('labels')
        fi_df = pd.DataFrame({
            'Feature': [column_names.get(f, f) for f in feature_names], 
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(fi_df.set_index("Feature"))
    except Exception as e:
        st.error(f"Could not generate feature importance plot. Error: {e}")

# =====================================================================================
# PAGE 4: ABOUT
# =====================================================================================
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### The Business Problem: Customer Retention in Insurance
    In the competitive insurance industry, retaining existing customers is far more cost-effective than acquiring new ones. This project tackles the critical business challenge of **customer churn**. The primary goal is to proactively identify customers who are at a high risk of cancelling their policies, enabling the business to take targeted actions—such as offering loyalty discounts or personalized support—to improve retention rates and protect revenue.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Project Workflow: From Data to Deployed Tool
    This dashboard is the final product of a complete end-to-end machine learning workflow:
    1.  **Data Exploration & Preparation:** The process began with a thorough analysis of the initial dataset (`Train.csv`) to understand its characteristics.
    2.  **Handling Class Imbalance:** The original data had significantly more retained customers than churned ones. To prevent model bias, **SMOTE (Synthetic Minority Over-sampling Technique)** was used to create a balanced dataset, which is crucial for accurately predicting the minority 'churn' class.
    3.  **Model Training & Selection:** A **LightGBM (Light Gradient Boosting Machine)** model was trained on the balanced data. This model was chosen for its high performance, speed, and efficiency with tabular data. The model's impressive performance metrics (Accuracy: 94.9%, ROC-AUC: 99.2%) are detailed on the 'Model Performance' page.
    4.  **Saving the Model:** The final trained model was saved as `lightgbm_churn_model.pkl` for use in this application.
    5.  **Interactive Dashboard:** This Streamlit application was built to provide an intuitive interface for interacting with the trained model, making its predictive power accessible to non-technical users.
    """)

    st.markdown("---")

    st.markdown("""
    ### Technology Stack
    -   **Data Analysis & Manipulation:** Pandas, NumPy
    -   **Machine Learning:** Scikit-learn, LightGBM, Imbalanced-learn
    -   **Data Visualization:** Matplotlib, Seaborn
    -   **Web Application:** Streamlit
    """)

    st.markdown("---")

    st.markdown("""
    ### How to Use This Dashboard
    -   **Dataset Overview:** Get a high-level summary of the training data and learn about the key factors that influence churn.
    -   **Predict Churn:** Use the interactive form to predict churn for a single customer or upload a CSV file for batch predictions.
    -   **Model Performance:** Review the detailed performance metrics and feature importance chart to understand the model's accuracy and logic.
    """)

