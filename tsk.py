import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

# Load the dataset
file_path = "dataset.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Select only the Soil Moisture column
soil_moisture = df[['Soil Moisture']].values

# Try different numbers of clusters (from 1 to 10)
bic_scores = []
aic_scores = []
n_clusters_range = range(1, 11)

for n in n_clusters_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(soil_moisture)
    bic_scores.append(gmm.bic(soil_moisture))
    aic_scores.append(gmm.aic(soil_moisture))

# Find the optimal number of clusters (lowest BIC score)
optimal_clusters = n_clusters_range[np.argmin(bic_scores)]
print(f"Optimal number of clusters: {optimal_clusters}")

# Plot BIC Score
plt.figure(figsize=(10, 4))
plt.plot(n_clusters_range, bic_scores, marker='o', linestyle='-', color='blue', label='BIC Score')
plt.xlabel("Number of Clusters")
plt.ylabel("BIC Score")
plt.title("Optimal Number of Clusters using BIC")
plt.legend()
plt.show()

# Plot AIC Score
plt.figure(figsize=(10, 4))
plt.plot(n_clusters_range, aic_scores, marker='s', linestyle='-', color='red', label='AIC Score')
plt.xlabel("Number of Clusters")
plt.ylabel("AIC Score")
plt.title("Optimal Number of Clusters using AIC")
plt.legend()
plt.show()

# Apply GMM with the optimal number of clusters
gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
df['Cluster'] = gmm.fit_predict(soil_moisture)

# Get soft clustering probabilities
probabilities = gmm.predict_proba(soil_moisture)
prob_cols = [f'Cluster_{i}' for i in range(optimal_clusters)]
df[prob_cols] = probabilities

# Sort values for smooth curve visualization
df_sorted = df.sort_values(by="Soil Moisture")

# Plot soft clustering probabilities
plt.figure(figsize=(10, 6))
for i in range(optimal_clusters):
    sns.lineplot(x=df_sorted["Soil Moisture"], y=df_sorted[f"Cluster_{i}"], label=f"Cluster {i}")

plt.xlabel("Soil Moisture")
plt.ylabel("Cluster Probability")
plt.title(f"GMM Soft Clustering (Optimal Clusters = {optimal_clusters})")
plt.legend()
plt.show()
