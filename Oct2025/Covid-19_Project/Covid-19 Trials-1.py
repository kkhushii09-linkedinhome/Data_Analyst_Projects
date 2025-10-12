import pandas as pd
df = pd.read_csv(r'C:\Users\kkhus\Downloads\UnifiedMentor\Covid 19 Clinical Trials\COVID clinical trials.csv')

print(df.head())

#check data types
print(df.info())

# Summary statistics for numerical columns
print(df.describe())

# Summary statistics for categorical columns
print(df.describe(include='object'))

# Check for missing values
print(df.isnull().sum())

# Drop columns with a high percentage of missing values or fill them
df = df.drop(columns=['Acronym', 'Study Documents']) # Example of dropping columns
df['Results First Posted'].fillna('Unknown', inplace=True)

print(df['Status'].value_counts())
df['Status'].value_counts().plot(kind='bar', title='Status of Clinical Trials')

print(df['Phases'].value_counts())
df['Phases'].value_counts().plot(kind='bar', title='Distribution of Phases')

import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV
df = pd.read_csv(r"C:\Users\kkhus\Downloads\UnifiedMentor\Covid 19 Clinical Trials\COVID clinical trials.csv")

# Convert Age column to numeric (force non-numeric to NaN)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Drop rows where Age is missing or invalid
df = df.dropna(subset=['Age'])

# Define bins and labels for age groups
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+']

# Create a new column for Age Groups
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Count number of people in each age group
age_group_counts = df['Age_Group'].value_counts().sort_index()
print("\nAge group distribution:\n", age_group_counts)

# Plot the distribution
plt.figure(figsize=(10,6))
sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='viridis')
plt.title("Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Count of People")
plt.show()

#phases vs status analysis

# Check if 'phase' and 'status' columns exist
if 'Phases' in df.columns and 'Status' in df.columns:
    # Replace missing or blank values
    df['Phases'] = df['Phases'].fillna('Unknown')
    df['Status'] = df['Status'].fillna('Unknown')

    # Create a cross-tabulation (contingency table)
    phase_status = pd.crosstab(df['Phases'], df['Status'])
    print("\nStatus vs Phase Table:\n", phase_status)

    # Plot a grouped bar chart
    phase_status.plot(kind='bar', figsize=(12,6))
    plt.title("COVID-19 Clinical Trials: Status vs Phase")
    plt.xlabel("Phase")
    plt.ylabel("Number of Trials")
    plt.legend(title="Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Optional: Create a heatmap for better visualization
    plt.figure(figsize=(10,6))
    sns.heatmap(phase_status, annot=True, fmt="d", cmap="viridis")
    plt.title("Heatmap: COVID-19 Clinical Trials Status vs Phase")
    plt.xlabel("Status")
    plt.ylabel("Phase")
    plt.show()
else:
    print("⚠️ Please check: Your dataset must contain 'phase' and 'status' columns.")

import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv(r"C:\Users\kkhus\Downloads\UnifiedMentor\Covid 19 Clinical Trials\COVID clinical trials.csv")
# Clean column names (in case there are extra spaces or uppercase)

print(df.columns) 
#Show only top 10 outcome measures per top 10 conditions

# Use exact column names as in your dataset
# Clean up columns
df['Conditions'] = df['Conditions'].fillna('Unknown')
df['Outcome Measures'] = df['Outcome Measures'].fillna('Unknown')

# Expand comma-separated lists
df_expanded = df.assign(
    Conditions=df['Conditions'].str.split(','),
    Outcome_Measures=df['Outcome Measures'].str.split(',')
).explode('Conditions').explode('Outcome_Measures')

# Clean text
df_expanded['Conditions'] = df_expanded['Conditions'].str.strip()
df_expanded['Outcome_Measures'] = df_expanded['Outcome_Measures'].str.strip()

# Build cross-tab
cond_outcome_ct = pd.crosstab(df_expanded['Conditions'], df_expanded['Outcome_Measures'])

# Take top 10 conditions and top 10 outcome measures
top_conditions = cond_outcome_ct.sum(axis=1).sort_values(ascending=False).head(10).index
top_outcomes = cond_outcome_ct.sum(axis=0).sort_values(ascending=False).head(10).index
cond_outcome_filtered = cond_outcome_ct.loc[top_conditions, top_outcomes]

# Plot clean heatmap
plt.figure(figsize=(10,6))
sns.heatmap(cond_outcome_filtered, cmap='viridis', annot=True, fmt='d')
plt.title("Top 10 Conditions vs Top 10 Outcome Measures (COVID-19 Trials)")
plt.xlabel("Outcome Measures")
plt.ylabel("Conditions")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Display available date-like columns
print("Date columns:", [col for col in df.columns if 'date' in col.lower()])

# Try to find and parse the start date
date_cols = [col for col in df.columns if 'date' in col.lower()]
if len(date_cols) > 0:
    # Assume 'Start Date' is the most relevant
    df['Start Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
else:
    print("No date columns found.")
    
# Drop missing dates
df = df.dropna(subset=['Start Date'])

# Extract Year-Month
df['Month'] = df['Start Date'].dt.to_period('M').astype(str)

# Count number of trials per month
monthly_counts = df['Month'].value_counts().sort_index()

# Plot the trend
plt.figure(figsize=(10,5))
plt.plot(monthly_counts.index, monthly_counts.values, marker='o', linewidth=2)
plt.title("COVID-19 Clinical Trials Started Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Trials Started")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Sum each numeric column and sort descending
values = numeric_df.sum().sort_values(ascending=False)[:40]

# Create the barplot
plt.figure(figsize=(12, 6))
sns.barplot(x=values.index, y=values.values)
plt.xticks(rotation=45, ha='right')  # Rotate x labels for readability
plt.ylabel('Sum of values')
plt.title('Top 40 Numeric Columns by Sum')
plt.tight_layout()
plt.show()
missing_data = df.isnull().mean() * 100
def visualize_data(data , caption = '' , ylabel = 'Percentage of Mising Data'):
    # set figure size
    sns.set(rc={'figure.figsize':(15,8.27)})
    # make ticks vertical
    plt.xticks(rotation=90)
    fig = sns.barplot(x = data.keys()[:min(40 ,
len(data))].tolist() , y = data.values[: min(40 ,
len(data))].tolist()) \
    .set_title(caption)
    # set labels
    plt.ylabel(ylabel)
    plt.show()
visualize_data(missing_data, 'Percentage of missing data in each feature')

# We can extract a new feature form The Location which is the country where the study hold
countries = [ str(df.Locations.iloc[i]).split(',')[-1] for i in
range(df.shape[0])]
df['Country'] = countries

