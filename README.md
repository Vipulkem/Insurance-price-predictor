# Insurance-price-predictor

## Project Overview

This project is a complete **Medical Insurance Cost Prediction and Visualization System** built with **Streamlit** and **Machine Learning**. It provides two key capabilities:

1. **Exploratory Data Analysis (EDA)** of the medical insurance dataset.
2. **Prediction of medical insurance charges** using a trained **Linear Regression model**.

The system uses the popular **insurance.csv** dataset, which includes demographic and lifestyle features such as age, sex, BMI, children, smoker status, region, and medical charges. The dataset serves as the foundation for both visualization and predictive modeling.

---

## Key Features

### 1. Machine Learning Model

* The project trains a **Linear Regression** model using:

  * **Age**
  * **Sex**
  * **BMI**
  * **Number of children**
  * **Smoking status**
  * **Region**
* Data preprocessing steps include:

  * Conversion of categorical variables to numeric format.
  * Label encoding for region, sex, and smoker.
* The model is trained, evaluated, and saved as **insurance_model.pkl** for reuse in the Streamlit dashboard.
* R² scores are computed for both training and testing sets to measure model performance.

---

### 2. Streamlit Dashboard (medical_boat.py)

A comprehensive interactive interface is provided with three main sections:

#### **Home**

* Overview of dataset structure and statistics.
* Tab-based visualizations:

  * Age distribution
  * Gender distribution
  * BMI distribution
  * Smoker distribution
  * Children distribution
  * Region distribution

#### **EDA (Exploratory Data Analysis)**

Two types of analysis are included:

##### **Bivariate Analysis**

* Age vs Gender
* Age vs Charges
* Smokers vs Gender
* BMI vs Charges
* Smokers vs Region
* Smoker vs Charges
* Each visualization includes insights and interpretations.

##### **Multivariate Analysis**

* Correlation heatmap
* BMI vs Age vs Charges (interactive Plotly scatter)
* Age vs Charges vs Smoker
* 3D scatter plot (age, BMI, charges)
* Pair plot (scatter matrix)

These visualizations help users understand how different variables affect medical charges.

#### **Insurance Price Predictor**

A user-friendly form to input:

* Age
* Gender
* BMI
* Children
* Smoker status
* Region

After submitting the form, the system uses the pre-trained model to estimate the **expected medical insurance cost**, shown instantly on the screen.

---

## Project Structure

```
├── insurance.csv              # Dataset
├── insurance_model.pkl        # Saved ML model
├── main.py                    # Model training and exporting script
├── medical_boat.py            # Streamlit dashboard + EDA + Prediction UI
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## Technologies Used

* **Python**
* **Streamlit** for UI and dashboards
* **Pandas, NumPy** for data processing
* **Matplotlib, Seaborn, Plotly** for visualizations
* **Scikit-learn** for machine learning
* **Pickle** for model serialization

---

## How It Works

1. Run **main.py**

   * Cleans and preprocesses the data
   * Trains the Linear Regression model
   * Evaluates performance
   * Saves model as *insurance_model.pkl*

2. Run **medical_boat.py**

   * Loads the dataset and the saved ML model
   * Displays interactive EDA dashboard
   * Provides insurance cost prediction tool

---

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the dashboard:

```
streamlit run medical_boat.py
```

---

## Purpose

This project can serve as:

* A learning tool for understanding regression modeling.
* An interactive platform for exploring insurance data.
* A practical example of deploying machine learning in a web-based interface.
* A foundation for building more advanced predictive healthcare systems.

