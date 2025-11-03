import pandas as pd

df=pd.read_csv(r'C:\Users\kkhus\Downloads\UnifiedMentor\Cyber Security\CloudWatch_Traffic_Web_Attack.csv')
df.info()
df.head()

# Check for missing values
missing_values = df.isnull().sum()

# Fill or drop missing values as needed
df['bytes_in'].fillna(df['bytes_in'].median(), inplace=True)
df.dropna(subset=['src_ip', 'dst_ip'], inplace=True)

# Convert columns to appropriate datatypes
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate total bytes by timestamp
time_group = df.groupby('creation_time')[['bytes_in', 'bytes_out']].sum()

# Plot the line chart
plt.figure(figsize=(12,5))
plt.plot(time_group.index, time_group['bytes_in'], label='Bytes In')
plt.plot(time_group.index, time_group['bytes_out'], label='Bytes Out')

plt.title("Bytes In and Bytes Out Over Time")
plt.xlabel("Time")
plt.ylabel("Total Bytes")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Count protocol occurrences
protocol_counts = df['protocol'].value_counts()

# Plot column chart
plt.figure(figsize=(8,5))
plt.bar(protocol_counts.index, protocol_counts.values)

plt.title("Count of Protocols Used")
plt.xlabel("Protocol")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Duration of the session in seconds
df['session_duration'] = (df['end_time'] -
df['creation_time']).dt.total_seconds()

# Average packet size
df['avg_packet_size'] = (df['bytes_in'] + df['bytes_out']) / df['session_duration']

# Count values by country code
country_counts = df['src_ip_country_code'].value_counts()

# Plot Pie Chart with country names as labels
plt.figure(figsize=(10, 8))
plt.pie(
    country_counts.values,
    labels=[f"{country} ({count})" for country, count in country_counts.items()],
    autopct='%1.1f%%',
    startangle=90
)

plt.title('Interaction Distribution by Source IP Country Code')
plt.axis('equal')  # Keeps pie chart circular
plt.tight_layout()
plt.show()

from sklearn.ensemble import IsolationForest
# Selecting features for anomaly detection
features = df[['bytes_in', 'bytes_out', 'session_duration',
'avg_packet_size']]
# Initialize the model
model = IsolationForest(contamination=0.05, random_state=42)
# Fit and predict anomalies
df['anomaly'] = model.fit_predict(features)
df['anomaly'] = df['anomaly'].apply(lambda x: 'Suspicious' if x
== -1 else 'Normal')

# Check the proportion of anomalies detected
print(df['anomaly'].value_counts())
# Display anomaly samples
suspicious_activities = df[df['anomaly'] == 'Suspicious']
print(suspicious_activities.head())

# Assuming your dataframe is named df and contains:
# timestamp, bytes_in, bytes_out

# Convert timestamp to datetime if needed
df['time'] = pd.to_datetime(df['time'])

# Sort by time to ensure correct order
df = df.sort_values('time').reset_index(drop=True)

# Define anomaly threshold using mean + 2*std
threshold_in = df["bytes_in"].mean() + 2 * df["bytes_in"].std()
threshold_out = df["bytes_out"].mean() + 2 * df["bytes_out"].std()

# Identify anomalies
anomalies_in = df["bytes_in"] > threshold_in
anomalies_out = df["bytes_out"] > threshold_out

plt.figure(figsize=(16, 7))

# Stacked bars: normal values
plt.bar(df.index, df["bytes_in"],
        label="Normal Bytes In", color="skyblue")
plt.bar(df.index, df["bytes_out"],
        bottom=df["bytes_in"],
        label="Normal Bytes Out", color="lightgreen")

# Highlight anomalies separately
plt.bar(df.index[anomalies_in], df["bytes_in"][anomalies_in],
        color="red", label="Anomaly Bytes In")
plt.bar(df.index[anomalies_out], df["bytes_out"][anomalies_out],
        bottom=df["bytes_in"][anomalies_out],
        color="orange", label="Anomaly Bytes Out")

grouped = df.groupby(['src_ip_country_code', 'detection_types']).size().unstack(fill_value=0)

# Plot
plt.figure(figsize=(14, 7))
grouped.plot(kind='bar', figsize=(14, 7))

plt.title('Detection Types by Source Country Code')
plt.xlabel('Source Country Code')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Detection Types')
plt.tight_layout()
plt.show()

# Convert timestamp field to datetime
df['creation_time'] = pd.to_datetime(df['creation_time'])

# Sort by time in case records are shuffled
df = df.sort_values('creation_time')

# Aggregate traffic over time (hourly granularity)
traffic_trend = df.resample('H', on='creation_time')[['bytes_in', 'bytes_out']].sum()

# Plot line chart for inbound and outbound traffic
plt.figure(figsize=(14, 6))
plt.plot(traffic_trend.index, traffic_trend['bytes_in'], marker='o', label='Bytes In')
plt.plot(traffic_trend.index, traffic_trend['bytes_out'], marker='o', label='Bytes Out')

plt.title('Web Traffic Analysis Over Time')
plt.xlabel('Time')
plt.ylabel('Total Bytes')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add edges from source IP to destination IP
for idx, row in df.iterrows():
    if pd.notna(row['src_ip']) and pd.notna(row['dst_ip']):  
        G.add_edge(row['src_ip'], row['dst_ip'])

# Draw the network graph
plt.figure(figsize=(14,10))
pos = nx.spring_layout(G, k=0.6, seed=42)  # Force-directed layout improves readability

nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue')
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

plt.title("Network Interaction between Source & Destination IPs")
plt.axis("off")
plt.show()

plt.figure(figsize=(14, 10))
nx.draw_networkx(G, with_labels=True, node_size=20,
font_size=8, node_color='skyblue', font_color='darkblue')
plt.title('Network Interaction between Source and Destination IPs')

s