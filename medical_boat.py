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

    page = st.sidebar.radio("Go to Page:",["Home","EDA","Insurance Price"])

    if page == "Home":
        st.title("Medical Insurance Data Visualization")
        st.write("Explore the insights about medical insurance costs.")
    

        tabs = st.tabs(["Stastical Overview", "Age Distribution","Gender Distribution","BMI Distribution","Smokers Distribution","Children Distribution","Regions Distribution"])

        with tabs[0]:
            st.header("Insurance Data Statistical Overview")
            st.write("----------------------------------------------------------------------")
            st.subheader("Data Overview")
            st.write("Here is the some statistic about the insurance data ")
            st.dataframe(Insurance_Data.head(10))
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
            st.header("Gender Distribution of Insurance Holders")

            counts = Insurance_Data['sex'].value_counts()

            # Create a new figure BEFORE plotting
            fig, ax = plt.subplots()
            ax.pie(
                counts,
                labels=['Male', 'Female'],
                autopct='%1.1f%%',
                colors=['lightblue', 'lightpink'],
                startangle=140
            )
            ax.set_title("Gender Distribution")

            st.pyplot(fig)     # Display in Streamlit

            # Separate figure for Age Distribution
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(Insurance_Data['age'], bins=20)
            ax2.set_title("Age Distribution")

            st.pyplot(fig2)



        with tabs[3]:
            st.header("BMI distribution of Insurance Holders")
            plt.figure(figsize=(5,3))
            sns.histplot(Insurance_Data['bmi'], bins=30, kde=True, color='Salmon')
            plt.title('BMI Distribution of Insurance Holders')
            plt.show()
            st.pyplot(plt)

            st.write("----------------------------------------------------------------------")

            st.subheader("Insights:")
            st.markdown('''
                        - The BMI distribution of insurance holders is right-skewed, with a significant number of individuals having a BMI between 20 and 30.
                        - There is a noticeable peak around the BMI range of 25-30, indicating a higher prevalence of overweight individuals among insurance holders.
                        - The distribution suggests that while many holders fall within the normal weight range, there is also a substantial portion that is overweight or obese.
                        - This information can be crucial for insurance companies in assessing risk and determining premiums based on health factors associated with BMI.
                        ''')
        with tabs[4]:
            st.header("Smoker vs Non-Smoker Insurance Holders")
            plt.figure(figsize=(5,3))
            sns.countplot(x='smoker', data=Insurance_Data, palette='Set2')
            plt.title('Smoker vs Non-Smoker Insurance Holders')
            plt.xticks(ticks=[0,1], labels=['Non-Smoker', 'Smoker'])
            plt.show()
            st.pyplot(plt)

            st.write("----------------------------------------------------------------------")

            st.subheader("Insights:")
            st.markdown('''
                        - The count plot reveals a significant disparity between the number of non-smokers and smokers among insurance holders.
                        - Non-smokers constitute a much larger portion of the insured population compared to smokers.
                        - This indicates that smoking status is a critical factor in insurance coverage, likely due to the associated health risks and higher medical costs for smokers.
                        - Insurance companies may use this information to tailor their policies and premiums based on smoking status.
                        ''')
            
        with tabs[5]:
            st.header("Children Distribution among Insurance Holders")
            plt.figure(figsize=(5,3))
            sns.countplot(x='children', data=Insurance_Data, palette='Set3')
            plt.title('Children Distribution among Insurance Holders')
            plt.show()
            st.pyplot(plt)

            st.write("----------------------------------------------------------------------")

            st.subheader("Insights:")
            st.markdown('''
                        - The count plot indicates that the majority of insurance holders have no children, followed by those with one or two children.
                        - There is a noticeable decrease in the number of holders as the number of children increases beyond two.
                        - This suggests that individuals with fewer or no children are more likely to have insurance coverage.
                        - The distribution may reflect financial considerations, as individuals with more children may face higher expenses and thus be less inclined to maintain insurance.
                        ''')

        with tabs[6]:
            st.header("Regions Distribution of Insurance Holders")
            plt.figure(figsize=(5,3))
            sns.countplot(x='region', data=Insurance_Data, palette='Set1')
            plt.title('Regions Distribution of Insurance Holders')
            plt.xticks(ticks=[0,1,2,3], labels=['Southwest', 'Southeast', 'Northwest', 'Northeast'])
            plt.show()
            st.pyplot(plt)

            st.write("----------------------------------------------------------------------")

            st.subheader("Insights:")
            st.markdown('''
                        - The count plot shows a relatively even distribution of insurance holders across the four regions: Southwest, Southeast, Northwest, and Northeast.
                        - However, the Southeast region has a slightly higher number of insurance holders compared to the other regions.
                        - This suggests that regional factors may influence insurance coverage, potentially due to differences in demographics, economic conditions, or healthcare access.
                        - Understanding regional distribution can help insurance companies tailor their marketing and service strategies to better meet the needs of specific areas.
                        ''')

    elif page == "EDA":
        st.title("Exploratory Data Analysis (EDA)")
        st.write("Dive deeper into the insurance data and uncover hidden insights.")

        tabs = st.sidebar.radio("Select Analysis:",["Bivariate Analysis","Correlation Heatmap"])

        if tabs == "Bivariate Analysis":

            tabs = st.tabs(["Age vs Gender","Age vs Charges","Smokers vs Gender","BMI vs Charges","Smokers vs Region","Smoker vs Charges"])      
        
            with tabs[0]:
                st.header("Age vs Gender of Insurance Holders")

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x="sex", data=Insurance_Data, hue="smoker", ax=ax)
                st.pyplot(fig)

            with tabs[1]:

                st.header("Age vs Charges Analysis")

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.regplot(x="age", y="charges", data=Insurance_Data, ax=ax, scatter_kws={'s': 50})

                ax.set_xlabel("Age")
                ax.set_ylabel("Charges")

                st.pyplot(fig)

                st.header("Age vs Charges by Smoker Status")

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x="age", y="charges", hue="smoker", data=Insurance_Data, ax=ax)

                st.pyplot(fig)
            with tabs[2]:

                st.header("Smokers vs Gender")

                fig, ax = plt.subplots(figsize=(8, 6))
                plot = sns.countplot(x="sex", hue="smoker", data=Insurance_Data, ax=ax)

                # Add count labels on each bar
                for p in plot.patches:
                    height = p.get_height()
                    ax.annotate(
                        str(height),
                        (p.get_x() + p.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )

                ax.set_xlabel("Sex")
                ax.set_ylabel("Count")

                st.pyplot(fig)
            
            with tabs[3]:

                st.header("BMI vs Charges Analysis")

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x="bmi", y="charges", hue="smoker", data=Insurance_Data, ax=ax)

                ax.set_xlabel("BMI")
                ax.set_ylabel("Charges")

                st.pyplot(fig)


                # st.header("BMI vs Charges (Average Charges per BMI Category)")

                # # Create BMI bins
                # Insurance_Data["bmi_group"] = pd.cut(
                #     Insurance_Data["bmi"],
                #     bins=[0, 18.5, 25, 30, 100],
                #     labels=["Underweight", "Normal", "Overweight", "Obese"]
                # )

                # bmi_charges = (
                #     Insurance_Data.groupby("bmi_group")["charges"]
                #     .mean()
                #     .reset_index()
                #     .rename(columns={"charges": "avg_charges"})
                # )

                # fig, ax = plt.subplots(figsize=(8, 6))
                # sns.barplot(x="bmi_group", y="avg_charges", data=bmi_charges, ax=ax)

                # # Add labels
                # for p in ax.patches:
                #     height = p.get_height()
                #     ax.annotate(
                #         f"{height:.0f}",
                #         (p.get_x() + p.get_width() / 2, height),
                #         ha="center",
                #         va="bottom"
                #     )

                # ax.set_xlabel("BMI Category")
                # ax.set_ylabel("Average Charges")

                # st.pyplot(fig)

            with tabs[4]:  

                st.header("Smokers vs Region Analysis")

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x="region", hue="smoker", data=Insurance_Data, ax=ax)

                ax.set_xlabel("Region")
                ax.set_ylabel("Count")

                st.pyplot(fig)

            with tabs[5]:

                st.header("Smoker vs Charges Analysis")

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x="smoker", y="charges", data=Insurance_Data, ax=ax)

                ax.set_xlabel("Smoker Status")
                ax.set_ylabel("Charges")

                st.pyplot(fig)


        elif tabs == "Correlation Heatmap":

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
            age = st.number_input("Age", min_value=18, max_value=80, value=20, step=1)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=20.0, step=0.1)
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
