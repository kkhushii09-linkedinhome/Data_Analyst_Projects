import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv(r'C:\Users\kkhus\Downloads\UnifiedMentor\DrugsAndEffects\drugs_side_effects_drugs_com.csv')
# Display the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# fill missing values with a placeholder
df_filled = df.fillna('Unknown')

# Distribution of drug ratings
plt.figure(figsize=(10, 6))
ax = sns.histplot(df['rating'], bins=10, kde=True)

# Add data labels to each bar
for patch in ax.patches:
    height = patch.get_height()
    if height > 0:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.5,
            int(height),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

plt.title('Distribution of Drug Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Count the most common drugs for each medical condition
top_drugs = df.groupby('medical_condition')['drug_name'].value_counts().nlargest(10)
print(top_drugs)

# Analyzing the most common side effects
side_effects = df['side_effects'].value_counts().head(10)
print(side_effects)

# Calculate average ratings per class
avg_ratings = df.groupby('drug_classes')['rating'].mean().sort_values(ascending=False)

# Plot column chart
plt.figure(figsize=(14, 8))
sns.barplot(x=avg_ratings.index, y=avg_ratings.values)

# Add data labels
for i, value in enumerate(avg_ratings.values):
    plt.text(i, value + 0.1, f"{value:.1f}", ha='center', fontsize=9)

plt.xticks(rotation=90)
plt.title("Average Drug Rating by Class")
plt.xlabel("Drug Class")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

print(df.columns.tolist())

# Group review counts
review_counts = df.groupby('drug_classes')['no_of_reviews'].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x=review_counts.index, y=review_counts.values)
plt.xticks(rotation=90)
plt.title("Total Number of Reviews by Drug Class")
plt.xlabel("Drug Class")
plt.ylabel("Review Count")

# Add labels
for i, v in enumerate(review_counts.values):
    plt.text(i, v + 50, str(int(v)), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))

# Bar plot for mean ratings
avg_ratings = df.groupby('drug_classes')['rating'].mean().sort_values(ascending=False)
sns.barplot(x=avg_ratings.index, y=avg_ratings.values, color='skyblue', alpha=0.7)

# Overlay boxplot
sns.boxplot(x='drug_classes', y='rating', data=df,
            order=avg_ratings.index, width=0.3, showfliers=False)

plt.xticks(rotation=90)
plt.title("Ratings Distribution by Drug Class (Box + Bar)")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# Compute average Rating
avg = df.groupby('drug_classes')['rating'].mean()

top5 = avg.sort_values(ascending=False).head(5)
bottom5 = avg.sort_values(ascending=True).head(5)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 5
sns.barplot(x=top5.values, y=top5.index, ax=axes[0], palette="Greens_r")
axes[0].set_title("Top 5 Drug Classes by Rating")
axes[0].set_xlabel("Average Rating")

# Bottom 5
sns.barplot(x=bottom5.values, y=bottom5.index, ax=axes[1], palette="Reds_r")
axes[1].set_title("Bottom 5 Drug Classes by Rating")
axes[1].set_xlabel("Average Rating")

plt.tight_layout()
plt.show()

# Categorize sentiment as None / Mild / Severe
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Filter positive and negative reviews
positive_reviews = df[df['rating'] >= 7]['side_effects'].dropna()
negative_reviews = df[df['rating'] < 7]['side_effects'].dropna()

# Prepare text for each category
positive_text = " ".join(positive_reviews).lower()
negative_text = " ".join(negative_reviews).lower()

# Define stopwords (add additional unnecessary words)
stopwords = set(STOPWORDS)
stopwords.update(["drug", "effects", "side", "symptoms", "also", "feel", "felt"])

# Create Word Clouds
plt.figure(figsize=(12, 6))

# Positive Word Cloud
plt.subplot(1, 2, 1)
wordcloud_pos = WordCloud(width=800, height=400, background_color="white",
                          stopwords=stopwords, colormap="Greens").generate(positive_text)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.title("Positive Reviews — Word Cloud")

# Negative Word Cloud
plt.subplot(1, 2, 2)
wordcloud_neg = WordCloud(width=800, height=400, background_color="white",
                          stopwords=stopwords, colormap="Reds").generate(negative_text)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.title("Negative Reviews — Word Cloud")

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset


# ✅ Clean & Filter data
df = df[(df['rating'].notna()) & (df['rating'] > 0)]
df['side_effects'] = df['side_effects'].fillna("")

# ==============================
# ✅ 1️⃣ Avg Rating by Drug Class — Bar Chart
# ==============================
rating_by_class = (
    df.groupby("drug_classes")["rating"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(14, 7))
sns.barplot(x=rating_by_class.values, y=rating_by_class.index)
plt.title("Average Rating by Drug Class")
plt.xlabel("Average Rating")
plt.ylabel("Drug Class")
plt.tight_layout()
plt.show()

# ==============================
# ✅ 2️⃣ Heatmap: Side-Effect Keywords vs Rating
# ==============================
keywords = ["nausea", "weight", "dizziness", "insomnia", "anxiety", "headache", "fatigue"]
for kw in keywords:
    df[kw] = df["side_effects"].str.contains(kw, case=False).astype(int)

corr_matrix = df[keywords + ["rating"]].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: Side-Effect Keywords vs Rating")
plt.show()

# ==============================
# ✅ 3️⃣ Side-Effect Severity vs Rating — Regression Chart
# ==============================
df["severity_score"] = df[keywords].sum(axis=1)

plt.figure(figsize=(8, 5))
sns.regplot(x=df["severity_score"], y=df["rating"])
plt.title("Side-Effect Severity vs Rating")
plt.xlabel("Severity (Count of Negative Side-Effects)")
plt.ylabel("Rating")
plt.show()

# ==============================
# ✅ 4️⃣ Sentiment Score Distribution (Rating-Based)
# ==============================
df["sentiment_label"] = df["rating"].apply(
    lambda r: "Positive" if r >= 7 else "Negative"
)

plt.figure(figsize=(8, 5))
sns.countplot(x="sentiment_label", data=df, palette="coolwarm")
plt.title("Sentiment Distribution Based on Ratings")
plt.xlabel("Sentiment")
plt.ylabel("Review Count")
plt.show()