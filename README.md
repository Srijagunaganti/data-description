
# Financial Risk Prediction for Luxury Fashion Companies

## Overview
This project aims to predict the financial risk for luxury fashion companies using machine learning. The financial risk is assessed using key financial indicators such as **Sales**, **Profit**, **COGS**, and **Gross Sales**. This project uses a **Random Forest Classifier** model to predict financial outcomes.

The code is intended to help businesses like **Hermès** and **Prada** understand the risks in their financial health, which can improve decision-making in terms of pricing, cost management, and profitability.

## Requirements

Before running the project, make sure you have the following Python libraries installed:

- **pandas**: for data manipulation.
- **matplotlib**: for creating visualizations.
- **seaborn**: for statistical data visualization.
- **scikit-learn**: for machine learning algorithms and metrics.

To install the required libraries, run the following command:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Project Structure

1. **financial_risk_prediction.py** – Python script containing the entire project code.
2. **Financials.csv** – Dataset with financial data (Sales, Profit, COGS, Gross Sales, etc.).

### Dataset

You will need to **download the dataset** (or ensure you have access to it). This dataset contains the financial data for companies (e.g., **Sales**, **Profit**, **COGS**, **Gross Sales**). You can place the dataset in the same directory as the Python script, and it will be automatically loaded when you run the code.

## Steps to Run the Code

### 1. Download the Dataset

- Download the **Financials.csv** dataset (make sure it has columns such as **Sales**, **Profit**, **COGS**, etc.).
- Place the **Financials.csv** file in the **same directory** as the Python script `financial_risk_prediction.py`.

### 2. Prepare the Python Script

- Open the Python script `financial_risk_prediction.py` in your preferred text editor or IDE.
- Make sure the following line correctly points to your dataset file:
  
    ```python
    file_path = r"C:\path\to\your\dataset\Financials.csv"
    ```

    Replace the path with the correct location of the `Financials.csv` dataset.

### 3. Run the Python Script

Once the dataset path is correctly set, you can run the Python script:

```bash
python financial_risk_prediction.py
```

### 4. Output

- The script will clean the data, perform exploratory data analysis (EDA), and generate the following visualizations:
  1. **Sales Distribution**: A histogram showing the distribution of sales in the dataset.
  2. **Correlation Matrix**: A heatmap showing correlations between key financial metrics like sales and profit.
  3. **Sales vs Profit**: A scatter plot illustrating the relationship between sales and profit.
  4. **Pairplot**: A pairwise plot of key financial features.
  
- The script will then train a **Random Forest Classifier** model to predict **Profit** based on other features.
- After training the model, the following will be printed:
  1. **Accuracy Score**: The accuracy of the model.
  2. **Confusion Matrix**: A heatmap showing how well the model performed in classifying financial risk.
  3. **Classification Report**: Precision, recall, and F1-score for the model.

### 5. Saving the Cleaned Dataset

- After cleaning the dataset, it will be saved to a file called **Cleaned_Financials.csv** in the same directory.
  
  You can open this CSV file to see the cleaned data that is ready for further analysis or model deployment.

---

## Troubleshooting

1. **Missing Libraries**: If you get an error like `ModuleNotFoundError`, make sure you've installed all the necessary libraries listed in the **Requirements** section.
   
2. **File Path Error**: If you encounter an error related to the dataset path, verify that you have correctly provided the full path to the **Financials.csv** dataset file.

3. **Data Formatting Issues**: If you have data formatting issues (e.g., currency symbols or commas not being removed), double-check the **currency_columns** list in the code. The dataset should have columns like **Manufacturing Price**, **Sale Price**, **Gross Sales**, **COGS**, and **Profit**.

---



