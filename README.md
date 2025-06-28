# Labour-Earning-Prediction
This project aims to predict an individual's earnings in 1978 based on demographic and historical income data. It uses machine learning to identify key factors influencing labor income, such as education, race, and marital status.
# 📈 Labour Earnings Prediction Project

This project aims to explore and model the relationship between various demographic and personal factors (such as education, race, marital status, and previous earnings) and a person's earnings in the year 1978. Using real-world labor data, we apply data analysis, preprocessing pipelines, and machine learning techniques to build a predictive model for income estimation. This can help understand the socio-economic dynamics and factors contributing to income inequality.

---

## 🧠 Objective

The primary objective of this project is to:
- Analyze key attributes influencing labor earnings.
- Preprocess and clean the dataset using modern machine learning pipelines.
- Train and evaluate regression models to predict `Earnings_1978`.
- Visualize patterns and compare model performance.

---

## 📂 Dataset

The dataset used is named `LabourTrainingData.csv`, which contains various columns such as:
- **Age**
- **Education**
- **Race**
- **Hispanic origin**
- **Marital Status**
- **Previous Earnings (1974, 1975)**
- **Nodeg (No Degree Indicator)**
- **Earnings in 1978 (Target)**

The dataset may include missing values and outliers, which are handled using imputation and scaling techniques.

---

## 🔧 Technologies & Libraries Used

- **Python**
- **Pandas, NumPy** – Data manipulation and transformation
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Preprocessing, modeling, and evaluation
- **Pipeline & ColumnTransformer** – For modular and reusable preprocessing
- **RandomForestRegressor** – Final model used for prediction

---

## 📊 Data Processing & EDA

The data preprocessing steps include:
- Fixing column names (e.g., correcting misspellings like "Eduacation")
- Handling missing values with `SimpleImputer`
- Scaling numerical features using `StandardScaler`
- Encoding categorical variables with `OneHotEncoder`
- Visualizing outliers using boxplots
- Grouping features by categories (Education, Race, Marital Status) for comparative analysis

---

## 🏗️ Modeling

We used the following steps to train our model:
- Split the dataset into training and testing sets
- Applied preprocessing pipeline using `ColumnTransformer`
- Trained multiple regression models (Random Forest, Gradient Boosting, Linear Regression)
- Evaluated the models using regression metrics like:
  - **R² Score**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**

The **Random Forest Regressor** performed best in terms of R² and generalization.

---

## 📈 Visualizations

Visual insights include:
- Bar plots of average earnings by education level, race, and marital status
- Boxplots for visualizing distribution and outliers
- Actual vs Predicted Earnings scatter plot for regression performance

---

## ✅ Results

- The model achieved a good **R² score** on the test set, indicating reliable prediction power.
- Key features affecting earnings included **Education**, **Marital Status**, and **previous earnings history**.
- Data imbalance and racial disparities were observed during exploratory analysis.

---

## 🔍 Future Improvements

- Implement cross-validation and hyperparameter tuning (e.g., `GridSearchCV`)
- Test additional models like XGBoost or CatBoost
- Export the model using `joblib` or `pickle` for deployment
- Build a web dashboard (e.g., with Streamlit or Flask) for real-time predictions

---

## 📁 Project Structure
📦 Labour_Earnings_Prediction/
│

├── Labour_Earning1.ipynb # Jupyter notebook with full workflow

├── LabourTrainingData.csv # Input dataset

├── README.md # Project documentation

└── requirements.txt # Python libraries to install
