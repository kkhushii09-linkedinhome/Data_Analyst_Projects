import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'C:\Users\kkhus\Downloads\UnifiedMentor\Covid 19 Clinical Trials\COVID clinical trials.csv')

def visualize_data(data, caption='', ylabel='Density'):
    # Set figure size
    plt.figure(figsize=(12, 6))
    
    # Generate different colors for each bar
    colors = sns.color_palette("Set2", n_colors=len(data))  # pick a palette
    
    # Create barplot
    ax = sns.barplot(x=data.index, y=data.values, palette=colors)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Set labels and title
    plt.ylabel(ylabel)
    plt.title(caption)
    
    # Add data labels on top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), ','),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=11)
    
    plt.tight_layout()
    plt.show()

status = df['Status'].value_counts()
visualize_data(status, caption='Status of The Application', ylabel='Density')

# Convert 'Start Date' to datetime
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

# Extract month name
start_month = df['Start Date'].dt.month_name()

# Count occurrences
start_month_distribution = start_month.value_counts()

# Visualize
visualize_data(start_month_distribution, caption='Start Month Distribution', ylabel='Density')