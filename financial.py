import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the data and models

def load_data_and_models():
    df = pd.read_csv("C:/Users/durga prasad/Downloads/previous_application.csv")  
    df2 = pd.read_csv("C:/Users/durga prasad/Downloads/application_data.csv")
    with open('C:/Users/durga prasad/Desktop/project/myenv/kmeans_model.pkl', 'rb') as file:
        risk_model = pickle.load(file)
    with open('C:/Users/durga prasad/Desktop/project/myenv/classifier.pkl', 'rb') as file:
        target_model = pickle.load(file)
    return df,df2, risk_model, target_model

df,df2, risk_model, target_model = load_data_and_models()
kmeans_model = risk_model[0]



def generate_mapping_for_column(df, column):
    # Convert values to strings
    df[column] = df[column].astype(str)
    # Get unique values and sort them
    unique_values = sorted(df[column].unique())
    # Create a mapping dictionary
    numerical_values = np.arange(0, len(unique_values))
    mapping = dict(zip(unique_values, numerical_values))
    return mapping

# Page layout
page = st.sidebar.radio("Go to", ('Introduction', 'Risk Detection', 'Target Detection'))

# Introduction page
if page == 'Introduction':
    st.title('Loan Risk and Target Detection')
    st.write('Welcome to the Loan Risk and Target Detection web app! Use the sidebar to navigate to different pages.')

# Risk Detection page
elif page == 'Risk Detection':
    st.title('Risk Detection')
    st.write('Enter the required information to detect risk.')

    # Add input widgets for required columns
    NAME_CONTRACT_TYPE = st.selectbox('Select NAME_CONTRACT_TYPE', options=df['NAME_CONTRACT_TYPE'].value_counts().index.sort_values())
    AMT_ANNUITY = st.number_input('AMT_ANNUITY')
    AMT_CREDIT = st.number_input('AMT_CREDIT')
    AMT_DOWN_PAYMENT = st.number_input('AMT_DOWN_PAYMENT')
    WEEKDAY_APPR_PROCESS_START = st.selectbox('Select WEEKDAY_APPR_PROCESS_START', options=df['WEEKDAY_APPR_PROCESS_START'].value_counts().index.sort_values())
    NAME_PAYMENT_TYPE = st.selectbox('Select NAME_PAYMENT_TYPE', options=df['NAME_PAYMENT_TYPE'].value_counts().index.sort_values())
    CODE_REJECT_REASON = st.selectbox('Select CODE_REJECT_REASON', options=df['CODE_REJECT_REASON'].value_counts().index.sort_values())
    NAME_TYPE_SUITE = st.selectbox('Select NAME_TYPE_SUITE', options=df['NAME_TYPE_SUITE'].value_counts().index.sort_values())
    NAME_CLIENT_TYPE = st.selectbox('Select NAME_CLIENT_TYPE', options=df['NAME_CLIENT_TYPE'].value_counts().index.sort_values())
    CNT_PAYMENT = st.number_input('CNT_PAYMENT')
    NAME_YIELD_GROUP = st.selectbox('Select NAME_YIELD_GROUP', options=df['NAME_YIELD_GROUP'].value_counts().index.sort_values())
    

    if st.button('Detect Risk'):
        if  NAME_CONTRACT_TYPE is not None:
            NAME_CONTRACT_TYPE_mapping=generate_mapping_for_column(df, 'NAME_CONTRACT_TYPE')
            if NAME_CONTRACT_TYPE in NAME_CONTRACT_TYPE_mapping:
                NAME_CONTRACT_TYPE=NAME_CONTRACT_TYPE_mapping[NAME_CONTRACT_TYPE]

        if WEEKDAY_APPR_PROCESS_START is not None:
            WEEKDAY_APPR_PROCESS_START_mapping=generate_mapping_for_column(df,'WEEKDAY_APPR_PROCESS_START')
        if WEEKDAY_APPR_PROCESS_START in WEEKDAY_APPR_PROCESS_START_mapping:
            WEEKDAY_APPR_PROCESS_START=WEEKDAY_APPR_PROCESS_START_mapping[WEEKDAY_APPR_PROCESS_START]

        if NAME_PAYMENT_TYPE is not None:
            NAME_PAYMENT_TYPE_mapping=generate_mapping_for_column(df,'NAME_PAYMENT_TYPE')
            if NAME_PAYMENT_TYPE in NAME_PAYMENT_TYPE_mapping:
                NAME_PAYMENT_TYPE=NAME_PAYMENT_TYPE_mapping[NAME_PAYMENT_TYPE]

        if CODE_REJECT_REASON is not None:
            CODE_REJECT_REASON_mapping=generate_mapping_for_column(df,'CODE_REJECT_REASON')
            if CODE_REJECT_REASON in CODE_REJECT_REASON_mapping:
                CODE_REJECT_REASON=CODE_REJECT_REASON_mapping[CODE_REJECT_REASON]

        if NAME_TYPE_SUITE is not None:
            NAME_TYPE_SUITE_mapping=generate_mapping_for_column(df,'NAME_TYPE_SUITE')
            if NAME_TYPE_SUITE in NAME_TYPE_SUITE_mapping:
                NAME_TYPE_SUITE=NAME_TYPE_SUITE_mapping[NAME_TYPE_SUITE]

        if NAME_CLIENT_TYPE is not None:
            NAME_CLIENT_TYPE_mapping=generate_mapping_for_column(df,'NAME_CLIENT_TYPE')
            if NAME_CLIENT_TYPE in NAME_CLIENT_TYPE_mapping:
                NAME_CLIENT_TYPE=NAME_CLIENT_TYPE_mapping[NAME_CLIENT_TYPE]

        if NAME_YIELD_GROUP is not None:
            NAME_YIELD_GROUP_mapping=generate_mapping_for_column(df,'NAME_YIELD_GROUP')
            if NAME_YIELD_GROUP in NAME_YIELD_GROUP_mapping:
                NAME_YIELD_GROUP=NAME_YIELD_GROUP_mapping[NAME_YIELD_GROUP]

        user_data = np.array([[NAME_CONTRACT_TYPE, AMT_ANNUITY, AMT_CREDIT, AMT_DOWN_PAYMENT,
                                WEEKDAY_APPR_PROCESS_START, NAME_PAYMENT_TYPE, CODE_REJECT_REASON,
                                NAME_TYPE_SUITE, NAME_CLIENT_TYPE, CNT_PAYMENT, NAME_YIELD_GROUP]])
        # Perform prediction using the risk model
        risk_prediction = kmeans_model.predict(user_data)
        st.success(f'Predicted Risk: {risk_prediction[0]}')


# Target Detection page
elif page == 'Target Detection':
    st.title('Target Detection')
    st.write('Enter the required information to detect loan payment frequency.')

    AMT_INCOME_TOTAL = st.number_input('AMT_INCOME_TOTAL')
    AMT_ANNUITY = st.number_input('AMT_ANNUITY')
    AMT_CREDIT = st.number_input('AMT_CREDIT')
    REGION_POPULATION_RELATIVE = st.number_input('REGION_POPULATION_RELATIVE')
    NAME_EDUCATION_TYPE = st.selectbox('Select NAME_EDUCATION_TYPE', options=df2['NAME_EDUCATION_TYPE'].value_counts().index.sort_values())
    OCCUPATION_TYPE = st.selectbox('Select OCCUPATION_TYPE', options=df2['OCCUPATION_TYPE'].value_counts().index.sort_values())
    ORGANIZATION_TYPE = st.selectbox('Select ORGANIZATION_TYPE', options=df2['ORGANIZATION_TYPE'].value_counts().index.sort_values())
    DAYS_EMPLOYED = st.number_input('DAYS_EMPLOYED')
    age = st.number_input('age')
    risk_segment = st.number_input('risk_segment')
    DAYS_REGISTRATION = st.number_input('DAYS_REGISTRATION')
    DAYS_ID_PUBLISH = st.number_input('DAYS_ID_PUBLISH')
    WEEKDAY_APPR_PROCESS_START = st.selectbox('Select WEEKDAY_APPR_PROCESS_START', options=df2['WEEKDAY_APPR_PROCESS_START'].value_counts().index.sort_values())

    if st.button('Target Predict'):
        if  NAME_EDUCATION_TYPE is not None:
            NAME_EDUCATION_TYPE_mapping=generate_mapping_for_column(df2, 'NAME_EDUCATION_TYPE')
            if NAME_EDUCATION_TYPE in NAME_EDUCATION_TYPE_mapping:
                NAME_EDUCATION_TYPE=NAME_EDUCATION_TYPE_mapping[NAME_EDUCATION_TYPE]

        if WEEKDAY_APPR_PROCESS_START is not None:
            WEEKDAY_APPR_PROCESS_START_mapping=generate_mapping_for_column(df2,'WEEKDAY_APPR_PROCESS_START')
            if WEEKDAY_APPR_PROCESS_START in WEEKDAY_APPR_PROCESS_START_mapping:
                WEEKDAY_APPR_PROCESS_START=WEEKDAY_APPR_PROCESS_START_mapping[WEEKDAY_APPR_PROCESS_START]

        if OCCUPATION_TYPE is not None:
            OCCUPATION_TYPE_mapping=generate_mapping_for_column(df2,'OCCUPATION_TYPE')
        if OCCUPATION_TYPE in OCCUPATION_TYPE_mapping:
            OCCUPATION_TYPE=OCCUPATION_TYPE_mapping[OCCUPATION_TYPE]

        if ORGANIZATION_TYPE is not None:
            ORGANIZATION_TYPE_mapping=generate_mapping_for_column(df2,'ORGANIZATION_TYPE')
            if ORGANIZATION_TYPE in ORGANIZATION_TYPE_mapping:
                ORGANIZATION_TYPE=ORGANIZATION_TYPE_mapping[ORGANIZATION_TYPE]
    
    
        user_data = np.array([[AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, NAME_EDUCATION_TYPE, REGION_POPULATION_RELATIVE, age,
                            DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, OCCUPATION_TYPE, WEEKDAY_APPR_PROCESS_START,
                            ORGANIZATION_TYPE, risk_segment]])
        # Perform prediction using the target model
        target_prediction = target_model.predict(user_data)
        st.write('Target Prediction:', target_prediction)