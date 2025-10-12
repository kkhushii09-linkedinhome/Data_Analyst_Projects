import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'C:\Users\kkhus\Downloads\UnifiedMentor\Covid 19 Clinical Trials\COVID clinical trials.csv')

df.Country.value_counts()[:35]

# Find the relation between null values in Acronym and Countries
(df.Acronym.isnull().groupby(df.Country).mean().sort_values(ascending = False) * 100)[:60]
missing_data = df.isnull().mean() * 100

def visualize_data(data , caption = '' , ylabel = 'Percentage of Mising Data'):
    # set figure size
    plt.figure(figsize=(15, 8))
    colors = sns.color_palette("tab10", n_colors=len(data))  # 10 colors for top 10
    ax = sns.barplot(x=data.index, y=data.values, palette=colors)
    
    # make ticks vertical
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(caption)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), ','),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
top_10_Countries = df['Country'].value_counts()[:10]
visualize_data(top_10_Countries, caption='Top 10 Countries', ylabel='Contributions')