import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = r"C:\Users\nicod\OneDrive\Desktop\Srija final project\Financials.csv"
df = pd.read_csv(file_path)

# Data Cleaning: Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Clean the currency columns by removing '$', ',', and converting them to numeric
currency_columns = ['Manufacturing Price', 'Sale Price', 'Gross Sales', 'COGS', 'Profit']

for col in currency_columns:
    df[col] = df[col].apply(lambda x: x.replace('$', '').replace(',', '') if isinstance(x, str) else x)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values by filling them with the mean of the respective columns
df.fillna(df.mean(), inplace=True)

# Exploratory Data Analysis (EDA)
# Set the visual style for better presentation
sns.set(style="whitegrid")

# 1. Distribution of 'Sales'
plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], kde=True, color='skyblue', bins=30)
plt.title('Sales Distribution', fontsize=14)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 2. Correlation Heatmap to see relationships between numeric columns
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.75})
plt.title('Correlation Matrix of Financial Features', fontsize=14)
plt.show()

# 3. Visualizing relationship between 'Sales' and 'Profit'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sales', y='Profit', data=df, color='green', alpha=0.7)
plt.title('Sales vs Profit', fontsize=14)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.show()

# 4. Pairplot of selected financial features
df_clean = df[['Sales', 'Profit', 'COGS', 'Gross Sales']].apply(pd.to_numeric, errors='coerce')
df_clean = df_clean.dropna()  # Remove any NaN values

sns.pairplot(df_clean, height=2.5, markers='o', hue='Profit', palette='coolwarm')
plt.suptitle('Pairplot of Key Financial Features', fontsize=14)
plt.show()

# Model Development
# Selecting 'Sales' as the target variable for prediction (you can change this based on your proposal's goal)
X = df.drop(columns=['Profit'])  # Features
y = df['Profit']  # Target variable

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model Evaluation
# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix - Predicted vs True Values', fontsize=14)
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Saving the cleaned data for further use (optional)
cleaned_file_path = r"C:\Users\nicod\OneDrive\Desktop\Srija final project\Cleaned_Financials.csv"
df_clean.to_csv(cleaned_file_path, index=False)

