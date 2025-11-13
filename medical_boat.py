import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')


Insurance_Data = pd.read_csv('insurance.csv')

# feature columns
categorical_features = []
Numeric_features = []

for column in Insurance_Data.columns:
    if Insurance_Data[column].dtype == 'object':
        categorical_features.append(column)
    else:
        Numeric_features.append(column)


Insurance_Data['sex'] = np.where(Insurance_Data['sex'] == 'male', 1, 0)
Insurance_Data['smoker'] = np.where(Insurance_Data['smoker'] == 'yes', 1, 0)
Insurance_Data.replace({'region': {'southwest':0, 'southeast':1, 'northwest':2, 'northeast':3}}, inplace=True)



with open('insurance_model.pkl', 'rb') as file:
    Linear_model = pickle.load(file)

st.set_page_config(page_title="Medical Insurance Coast Predictor", page_icon=":hospital:", layout="wide",initial_sidebar_state="collapsed")   

def dashoard():
    st.sidebar.title(f'Hello user!......')
    st.sidebar.markdown("--------------------------------")

    page = st.sidebar.radio("Go to Page:",["Home","Insurance Price"])

    if page == "Home":
        st.title("Medical Insurance Data Visualization")
        st.write("Explore the insights about medical insurance costs.")
    

        tabs = st.tabs(["Stastical Overview", "Age Distribution","Heatmap"])

        with tabs[0]:
            st.header("Insurance Data Statistical Overview")
            st.write("----------------------------------------------------------------------")
            st.subheader("Data Overview")
            st.write("Here is the some statistic about the insurance data ")
            st.dataframe(Insurance_Data)
            st.write("----------------------------------------------------------------------")
            st.dataframe(Insurance_Data.describe())

            st.subheader("Statistical Summary")
            st.write("----------------------------------------------------------------------")  
            st.write("Categorical Features  : " + ", ".join(categorical_features))
            st.write("Numeric Features: " + ", ".join(Numeric_features))        

        with tabs[1]:
            st.header("Age distribution of Insurance Holders")
            plt.figure(figsize=(5,3))
            sns.histplot(Insurance_Data['age'], bins=30, kde=True, color='green')
            plt.title('Age Distribution of Insurance Holders')
            plt.show()
            st.pyplot(plt)
            
            st.write("----------------------------------------------------------------------")

            st.subheader("Insights:")
            st.markdown('''
                        - The age distribution of insurance holders shows a peak in the 20-30 age range.    
                        - There is a gradual decline in the number of holders as age increases beyond 30.
                        - This suggests that younger individuals are more likely to have insurance coverage compared to older age groups.
                        - The presence of a few older holders indicates that some individuals maintain insurance into their later years.    
                        - Overall, the distribution highlights the importance of targeting younger demographics for insurance products.
                        ''')
        with tabs[2]:
            st.header("Correlation Heatmap of Insurance Data")
            plt.figure(figsize=(8,6))
            correlation_matrix = Insurance_Data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='icefire', fmt=".2f")
            plt.title('Correlation Heatmap')
            plt.show()
            st.pyplot(plt)

            st.write("----------------------------------------------------------------------")

            st.subheader("Insights:")
            st.markdown('''
                        - The heatmap reveals strong positive correlations between 'charges' and features like 'age', 'bmi', and 'smoker'.
                        - This indicates that older individuals, those with higher BMI, and smokers tend to incur higher insurance costs.
                        - Conversely, features like 'children' show weaker correlations with 'charges', suggesting a lesser impact on insurance costs.
                        - Understanding these relationships can help in risk assessment and pricing strategies for insurance providers.
                        ''')
        
    elif page == "Insurance Price":
        st.title("Insurance Price Predictor")
        st.write("--------------------------------")

        st.subheader("Enter the details to predict the insurance cost:")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            children = st.number_input("Number of Children", min_value=0, max_value=5, value=0, step=1)

        with col2:
            sex = st.selectbox("Gender",("Male","Female"))
            smoker = st.selectbox("Smoker",("Yes","No"))
            region = st.selectbox("Region",("Southwest","Southeast","Northwest","Northeast"))

        if st.button("Predict Insurance Cost"):
            sex = 1 if sex == "Male" else 0
            smoker = 1 if smoker == "Yes" else 0
            region_dict = {"Southwest":0, "Southeast":1, "Northwest":2, "Northeast":3}
            region = region_dict[region]

            input_data = np.array([[age,sex, bmi, children, smoker, region]])
    
            prediction_df = Linear_model.predict(input_data)[0]
            st.success(f"The predicted insurance cost is: ${prediction_df:.2f}")

        


    








dashoard()
