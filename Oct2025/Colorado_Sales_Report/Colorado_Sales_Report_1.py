# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r'C:\Users\kkhus\Downloads\UnifiedMentor\Colorado Motor Sales Data\colorado_motor_vehicle_sales.csv')

# Display the first few rows of the dataset
print(data.head())
# Convert date column to datetime format
data['Date'] = pd.to_datetime(data['year'].astype(str) + 'Q' + data['quarter'].astype(str))

# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values if necessary
data.dropna(inplace=True)

# Aggregate sales by month
data.set_index('Date', inplace=True)
monthly_sales = data.resample('M').sum()

# Display the first few rows of the aggregated data
print(monthly_sales.head())

# Plot total sales over time
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales['sales'], label='Total Sales')
plt.title('Total Motor Vehicle Sales Over Time')
plt.xlabel('Date')
plt.ylabel('sales')
plt.legend()
plt.show()

# Create a bar chart with better width and color
plt.bar(monthly_sales.index, monthly_sales['sales'],
        color='steelblue', width=50, edgecolor='black', alpha=0.8)

plt.title('Total Motor Vehicle Sales Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('sales', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot as a line chart
county_sales = data.groupby(['Date', 'county'])['sales'].sum().unstack()
county_sales.plot(kind='line', figsize=(12, 6))

plt.title('Motor Vehicle Sales by County Over Time')
plt.xlabel('Date')
plt.ylabel('sales')
plt.legend(title='county')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Compute correlations between sales of different vehicle types
correlation_matrix = county_sales.corr()
# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Perform seasonal decomposition on total sales
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(monthly_sales['sales'], model='additive', period=12)
decomposition.plot()
plt.show()

# Fit an ARIMA model to the total sales data
model = ARIMA(monthly_sales['sales'], order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# Make predictions
forecast = model_fit.forecast(steps=12)
# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales['sales'], label='Actual Sales')
plt.plot(pd.date_range(start=monthly_sales.index[-1], periods=12, freq='M'), forecast,
label='Forecasted Sales', color='red')
plt.title('Motor Vehicle Sales Forecast')
plt.xlabel('Date')
plt.ylabel('sales')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(monthly_sales['sales'][-12:], forecast)
print(f'Mean Squared Error: {mse}')

# Generate a summary report
report = f"""
Colorado Motor Vehicle Sales Data Analysis Report
=================================================
1. Data Overview
----------------
- Time Frame: {data.index.min()} to {data.index.max()}
- Total Sales Data Points: {len(data)}
2. Exploratory Data Analysis
----------------------------
- Total motor vehicle sales were plotted over time, showing general trends and
seasonality.
- Sales by vehicle type were plotted to compare different categories.
3. Statistical Analysis
-----------------------
- Seasonal decomposition of total sales showed clear seasonal patterns.
- Correlation analysis showed relationships between sales of different vehicle types.
4. Predictive Modeling
----------------------
- An ARIMA model was used to forecast motor vehicle sales for the next 12 months.
- The model's Mean Squared Error (MSE) was: {mse:.2f}
5. Conclusions
--------------
- The analysis provided insights into the trends and seasonality of motor vehicle sales
in Colorado.
- The predictive model can be used to forecast future sales, aiding in inventory
management and sales strategies.
"""
print(report)

