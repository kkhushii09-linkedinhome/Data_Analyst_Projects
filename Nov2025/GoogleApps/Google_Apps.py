import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv(r'C:\Users\kkhus\Downloads\googleplaystore.csv')

# Display the first few rows
print(df.head())

# Get basic information about the dataset
print(df.info())
# Summary statistics of numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())
# Handle missing values (e.g., filling or dropping)
df['Rating'].fillna(df['Rating'].mean(), inplace=True)
df.dropna(subset=['App', 'Category'], inplace=True)

# Handle missing values (e.g., filling or dropping)
df['Rating'].fillna(df['Rating'].mean(), inplace=True)
df.dropna(subset=['App', 'Category'], inplace=True)
# Convert columns to appropriate data types
def clean_reviews(x):
    x = str(x).strip()
    if 'M' in x:
        return int(float(x.replace('M','')) * 1_000_000)
    elif 'K' in x:
        return int(float(x.replace('K','')) * 1_000)
    else:
        return int(float(x))

df['Reviews'] = df['Reviews'].apply(clean_reviews)

# ---- CLEAN INSTALLS COLUMN ----
# Convert to string first
df['Installs'] = df['Installs'].astype(str)

# Remove bad rows like 'Free'
df = df[df['Installs'] != 'Free']

# Clean formatting
df['Installs'] = (
    df['Installs']
    .str.replace(',', '', regex=False)
    .str.replace('+', '', regex=False)
    .astype(int)
)

# ---- CLEAN PRICE COLUMN ----
df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False).astype(float)

# Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribution of App Ratings')
plt.show()

# Count of Apps by Category
plt.figure(figsize=(12, 8))
sns.countplot(y='Category', data=df,
order=df['Category'].value_counts().index)
plt.title('Count of Apps by Category')
plt.show()

# Relationship between Installs and Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating', y='Installs', hue='Category',
data=df)
plt.title('Relationship between Installs and Ratings')
plt.show()

# Average rating by category
avg_rating_by_category = df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
print(avg_rating_by_category)

# Most popular apps (by installs)
most_installed_apps = df[['App', 'Installs']].sort_values(by='Installs', ascending=False).head(10)
print(most_installed_apps)

# Top 5 genres
top_genres = df['Genres'].value_counts().head(5)
print(top_genres)

# Plot Missing Values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# make figure size
plt.figure(figsize=(16, 6))
# plot the null values by their percentage in each column
missing_percentage = df.isnull().sum()/len(df)*100
missing_percentage.plot(kind='bar')
# add the labels
plt.xlabel('Columns')
plt.ylabel('Percentage')
plt.title('Percentage of Missing Values in each Column')
numeric_cols = [i for i in df.columns if df[i].dtype != 'object' ]
if "Installs_category" in numeric_cols:
    numeric_cols.remove("Installs_category")
corr = df[numeric_cols].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.show()
from scipy import stats
df_clean = df.dropna()
# calculate Pearson's R between Rating and Installs
pearson_r, _ = stats.pearsonr(df_clean['Reviews'], df_clean['Installs'])
print(f"Pearson's R between Reviews and Installs: {pearson_r:.4f}")
df.dropna(subset=['Current Ver', 'Android Ver', 'Category', 'Type', 'Genres'],
inplace=True)

print(f"Length of the dataframe after removing null values: {len(df)}")

df.groupby('Installs')['Rating'].describe()

df.dropna(subset=['Current Ver', 'Android Ver', 'Category', 'Type', 'Genres'],
inplace=True)

print(f"Length of the dataframe after removing null values: {len(df)}")

df['Rating'].isnull().sum()

df['Installs'].loc[df['Rating'].isnull()].value_counts()

# plot the boxplot of Rating in each Installs_category
plt.figure(figsize=(16, 6)) # make figure size
sns.boxplot(x='Installs', y='Rating', hue='Installs', data=df)
df['Installs'] = pd.cut(
    df['Installs'],
    bins=[0, 1_000, 100_000, 1_000_000, 10_000_000, 1_000_000_000],
    labels=['1K', '100K', '1M', '10M', '1B']
)
def fill_missing_ratings(df, category, fill_value):

# Filter the DataFrame for rows where the category matches and rating is missing
    filtered_df = df[(df['Category'] == 'Category') & (df['Rating'].isnull())]
    df.loc[filtered_df.index, 'Rating'] = fill_value
    df['Rating'] = df.groupby('Category')['Rating'].transform(
        lambda x: x.fillna(x.mean())
)
    return df   

df = fill_missing_ratings(df, 'Low', 4.170970)

df = fill_missing_ratings(df, 'Very low', 4.637037)
df = fill_missing_ratings(df, 'Moderate', 4.035417)
df = fill_missing_ratings(df, 'More than moderate', 4.093255)
df = fill_missing_ratings(df, 'High', 4.207525)

df = fill_missing_ratings(df, 'no', 0)

df['Installs'].loc[df['Rating'].isnull()].value_counts()

plt.figure(figsize=(16, 6)) # make figure size
sns.boxplot(x='Installs', y= 'Reviews', data=df)
plt.figure(figsize=(16, 6)) # make figure size
sns.scatterplot(x='Reviews', y='Installs', data=df)


plt.figure(figsize=(16, 6)) # make figure size
sns.scatterplot(x=np.log10(df['Reviews']), y=np.log10(df['Installs']), data=df)

# let's plot the same plots for Reviews column as well
plt.figure(figsize=(16, 6)) # make figure size
sns.boxplot(x='Installs', y= np.log10(df['Reviews']), data=df) # plot theboxplot

df['Installs'].loc[df['Rating'].isnull()].value_counts()

df['Rating'].isnull().sum()

plt.figure(figsize=(16, 6)) # make figure size
sns.boxplot(x='Installs', y= 'Reviews', data=df)

print(df['Category'].value_counts().head(10))

