import numpy as np
import pandas as pd
data = pd.read_csv(r'C:\Users\kkhus\Downloads\UnifiedMentor\Colorado Motor Sales Data\colorado_motor_vehicle_sales.csv')

data.head()

import matplotlib.pyplot as plt
import seaborn as sns
def perform_eda(data):
# Print the shape of the DataFrame
    print(f"Shape of the DataFrame: {data.shape}\n")

    print(f"Data types:\n{data.dtypes}\n")

    # Check for missing values
    print(f"Missing values:\n{data.isnull().sum()}\n")

    # Summary statistics
    print(f"Summary statistics:\n{data.describe()}\n")

# For each column
for column in data.columns:
# Check if the column is numeric
    if pd.api.types.is_numeric_dtype(data[column]):
# Plot a histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(data=data, x=column, kde=True)
        plt.title(f"Histogram of {column}")
        plt.show()
    elif data[column].dtype == 'object':
# Plot a bar plot
        plt.figure(figsize=(6, 4))
        sns.countplot(data=data, x=column)
        plt.title(f"Bar plot of {column}")
        plt.xticks(rotation=90)
        plt.show()



# Create a new column that represents the year and quarter
data['period'] = data['year'].astype(str) + ' Q' + data['quarter'].astype(str)

# Time series plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='period', y='sales')
plt.title('Sales Over Time')
plt.xticks(rotation=90)
plt.show()

# Box plot by quarter
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='quarter', y='sales')
plt.title('Sales Distribution by Quarter')
plt.show()

# Get all unique years and quarters
years = sorted(data['year'].unique())
quarters = sorted(data['quarter'].unique())

# Loop through each year and quarter
for year in years:
    for quarter in quarters:
        # Filter the DataFrame
        filtered_df = data[(data['year'] == year) & (data['quarter'] == quarter)]
        
        # Skip if no data
        if filtered_df.empty:
            continue
        
        # Group by county and sum sales
        county_sales = filtered_df.groupby('county')['sales'].sum().reset_index()
        county_sales_sorted = county_sales.sort_values('sales', ascending=False)
        
        # Create the bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=county_sales_sorted, 
            x='county', 
            y='sales', 
            palette='viridis'
        )
        plt.title(f'Sales by County for {year} Q{quarter}')
        plt.xticks(rotation=90)
        plt.ylabel('Total Sales')
        plt.xlabel('County')
        plt.show()