#Load and inspect the data:
import pandas as pd
data=pd.read_csv(r"C:\Users\kkhus\Downloads\UnifiedMentor\Coffee-Sales\index_1.csv")
print(data.head)
 # Check for missing values
print(data.isnull().sum())
 



# Convert Date to datetime type
data['date'] = pd.to_datetime(data['date'], errors='coerce')
print(data['date'].head())
print(data['date'].isna().sum())

# Extract month and year from the Date
data['Month'] = data['date'].dt.month
data['Year'] = data['date'].dt.year
 # Drop the original Date column


import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean style
sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 7))

# Lineplot with markers and better palette
sns.lineplot(
    data=data,
    x='Month',
    y='Sales',
    hue='Year',
    marker="o",
    palette="tab10",
    linewidth=2.5
)

# Titles and labels
plt.title("📈 Monthly Sales Over Years", fontsize=18, weight='bold', pad=20)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Sales", fontsize=14)

# Move legend outside
plt.legend(title="Year", fontsize=12, title_fontsize=13, bbox_to_anchor=(1.05, 1), loc='upper left')

# Improve ticks
plt.xticks(range(1, 13), 
           ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 
           fontsize=12)
plt.yticks(fontsize=12)

# Add light grid
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
# Sales by store
# Set theme
sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 7))

# Sort stores by sales for cleaner display
sorted_data = data.groupby("Store", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)

# Barplot
ax = sns.barplot(
    data=sorted_data,
    x="Store",
    y="Sales",
    palette="Set2"
)

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='black')

# Title & labels
plt.title("🏬 Sales by Store", fontsize=18, weight="bold", pad=20)
plt.xlabel("Store", fontsize=14)
plt.ylabel("Sales", fontsize=14)

# Rotate store labels if needed
plt.xticks(rotation=30, ha="right", fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

 # Sales by product
# Set theme
sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 7))

# Sort products by sales for clarity
sorted_data = (
    data.groupby("coffee_name", as_index=False)["Sales"]
    .sum()
    .sort_values("Sales", ascending=False)
)

# Horizontal barplot
ax = sns.barplot(
    data=sorted_data,
    y="coffee_name",
    x="Sales",
    palette="crest"
)

# Add data labels
for p in ax.patches:
    ax.annotate(
        f'{p.get_width():,.0f}',  # format with commas
        (p.get_width(), p.get_y() + p.get_height() / 2),
        ha="left", va="center",
        fontsize=11, fontweight="bold",
        color="black", xytext=(5, 0),
        textcoords="offset points"
    )

# Titles and labels
plt.title("☕ Sales by Coffee Product", fontsize=18, weight="bold", pad=20)
plt.xlabel("Total Sales", fontsize=14)
plt.ylabel("Coffee Name", fontsize=14)

# Ticks styling
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
#monthly sales

monthly_sales = (
    data.groupby(['coffee_name', 'Month'])
        .count()['Sales']  # use 'Sales' instead of 'date'
        .reset_index()
        .rename(columns={'Sales':'count'})
        .pivot(index='Month', columns='coffee_name', values='count')
        .reset_index()
)
monthly_sales

monthly_sales.describe().T.loc[:,['min','max']]

plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_sales)
plt.legend(loc='upper left')
plt.xticks(range(len(monthly_sales['Month'])),monthly_sales['Month'],size='small')

# Extract weekday name
data['weekday'] = data['date'].dt.day_name()

#weekly sales
weekday_sales = (
    data.groupby('weekday')
        .count()['Sales']  # use a column that exists instead of 'date'
        .reset_index()
        .rename(columns={'Sales':'count'})
)

# Optionally, order weekdays
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales['weekday'] = pd.Categorical(weekday_sales['weekday'], categories=weekday_order, ordered=True)
weekday_sales = weekday_sales.sort_values('weekday')

weekday_sales

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12,6))

# Barplot
ax = sns.barplot(
    data=weekday_sales,
    x='weekday',
    y='count',
    color='steelblue'
)

# Add data labels
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():,}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=11, fontweight='bold'
    )

# Titles and labels
plt.title("📊 Sales by Weekday", fontsize=16, weight='bold', pad=15)
plt.xlabel("Weekday", fontsize=12)
plt.ylabel("Sales Count", fontsize=12)

# Rotate x-ticks if needed
plt.xticks(fontsize=12)

plt.tight_layout()
plt.show()

#Cash or Card
data['cash_type'].hist()

#coffee type proportions in sales hourly
def extract_hour(dt_str):
    try:
        mins, secs = map(float, dt_str.split(':'))
        total_hours = int(mins // 60)  # convert cumulative minutes to hours
        return total_hours
    except:
        return None
    
# Ensure Sales is numeric



# Drop invalid rows
hourly_data = data.dropna(subset=['Sales', 'hour'])
hourly_data['hour'] = hourly_data['hour'].astype(int)


# Group by hour and sum Sales
hourly_sales = hourly_data.groupby('hour')[['Sales']].sum().reset_index()
hourly_sales

#show hourly sales chart
plt.figure(figsize=(12,6))

# Barplot
ax = sns.barplot(data=hourly_sales, x='hour', y='Sales', color='steelblue')

# Add data labels
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.0f}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=11, fontweight='bold'
    )

plt.title("🕒 Hourly Sales", fontsize=16, weight='bold')
plt.xlabel("Hour of Day")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Hourly Sales by Coffee Type
hourly_sales_by_coffee=data.groupby(['hour','coffee_name']).count()['date'].reset_index().rename(columns={'date':'count'}).pivot(index='hour',
 columns='coffee_name',values='count').fillna(0).reset_index()
hourly_sales_by_coffee
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

 # Flatten the array of subplots for easy iteration
axs = axs.flatten()

 # Loop through each column in the DataFrame, skipping the'Index' column
for i, column in enumerate(hourly_sales_by_coffee.columns[1:]):
 # Skip the first column ('Index')
    axs[i].bar(hourly_sales_by_coffee['hour'],
    hourly_sales_by_coffee[column])
    axs[i].set_title(f'{column}')
    axs[i].set_xlabel('Hour')
 #axs[i].set_ylabel('Sales')
plt.tight_layout()
 # Show the plot
plt.show()

from prophet import Prophet

# Prepare data
df_prophet = data.groupby('date')['Sales'].sum().reset_index()
df_prophet.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y'

# Initialize model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Fit model
model.fit(df_prophet)

# Make future dataframe (next 30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# Show next day forecast
print("Next Day Forecasted Sales:", forecast['yhat'].iloc[-1])





import datetime as dt

# Ensure 'date' is datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Latest date in dataset
snapshot_date = data['date'].max() + pd.Timedelta(days=1)

# Aggregate per user
rfm = data.groupby('user_id').agg({
    'date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Sales': ['count','sum']  # Frequency and Monetary
})

# Flatten column names
rfm.columns = ['Recency','Frequency','Monetary']
print(rfm.head())

# Optional: show summary statistics per segment
segment_summary = rfm.groupby('Segment').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':'mean',
    'Segment':'count'
}).sort_values('Monetary', ascending=False)

print(segment_summary)
# Optional: preview
rfm.head()
from sklearn.cluster import KMeans

# Find optimal clusters (optional)
import matplotlib.pyplot as plt

inertia = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

plt.plot(range(2,10), inertia, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# 2D Scatter of Frequency vs Monetary colored by Segment
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=rfm['Frequency'],
    y=rfm['Monetary'],
    hue=rfm['Segment'],
    palette='tab10'
)
plt.title("Customer Segments: Frequency vs Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.show()s

# Fit final model (example with 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Check segments
rfm.groupby('Segment').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':'mean',
    'Segment':'count'
}).sort_values('Monetary', ascending=False)
