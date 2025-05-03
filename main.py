# main

# imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

# Load the CSV file (all rows are treated as data)
data = pd.read_csv("tokenizer-table.csv", header=None)
#print("Data shape:", data.shape)

# Extract the word names from the first row; first cell is empty so we skip it
theWords = data.iloc[0, 1:].tolist()

# Create column names: label the first column as 'ID' and the rest as token names
headers = ['ID'] + theWords

#print("Data shape:", data.shape)

# Select the rows corresponding to actual document data.
# Rows 0 to 2 are header, Total, and Average respectively.
df_texts = data.iloc[3:].copy()

# Assign our headers to these rows
df_texts.columns = headers

# Define a function to split the document ID (e.g., "FED_18_C") into paper number and author code.
def parse_id(doc_id):
    # Split the string using '_' as the delimiter
    parts = doc_id.split('_')  # Expected output: ['JA', 'prideandprejudice']
    # Check and extract the paper number and author code
    if len(parts) == 2:
        author = parts[0]
        title = parts[1]
    else:
        author = None
        title = None
    return pd.Series([author, title], index=['Author', 'Title'])

# Apply the function to the 'ID' column to create two new columns: 'Author' and 'Title'
df_texts[['Author', 'Title']] = df_texts['ID'].apply(parse_id)

# Display the updated DataFrame
#print(df_texts.info())
#print(df_texts.head())
#print(df_texts.tail())


# create array of features, except the initial (class) column
features = df_texts.columns
features = features[1:-2] # remove ID, Author, Title

# create dataframe of features (X) and classes (y)
X = df_texts[features].values
# ——— Fix #1: Standardize each token’s frequency across documents ———
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df_texts['Author'].values

classes = []
for label in y:
    classes.append(label)
    
y = pd.DataFrame(y)
y = y.values.tolist()

uniqueClasses = []
for each in classes:
    if each not in uniqueClasses:
        uniqueClasses.append(each)

print ("Classes:", classes)
print("Unique Classes:", uniqueClasses)

# Reduce the dimensionality to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Optional: view the amount of variance explained by the principal components
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Determine the number of unique known authors
n_clusters = len(uniqueClasses)
print("Number of clusters set to:", n_clusters)

# Run K-means on the scaled feature matrix
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Create a temporary DataFrame to view the clusters with actual authors.
temp_df = pd.DataFrame({
    'Cluster': clusters,
    'Author': df_texts['Author']
})

# Determine the majority (most common) author in each cluster.
cluster_to_author = temp_df.groupby('Cluster')['Author'].agg(lambda x: x.value_counts().index[0]).to_dict()
print("Mapping from Cluster to Author:", cluster_to_author)

# OPTIONAL: Use the PCA-reduced data for plotting.
# Here, X_pca is the 2-component PCA result of your standardized features.
# You should have already computed X_pca from your earlier steps.

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k') # s = (some #) to have a seed
plt.title("K-Means Clustering of Jane Austen and GPT4o")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Create legend items for each cluster.
handles = []
# Get the colors from the colormap that is used by the scatter plot.
# np.linspace ensures we pick colors uniformly for each cluster.
colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(cluster_to_author)))
for cluster, color in zip(sorted(cluster_to_author.keys()), colors):
    author_label = cluster_to_author[cluster]
    patch = mpatches.Patch(color=color, label=f"Cluster {cluster} : {author_label}")
    handles.append(patch)

# Add the legend to the plot.
plt.legend(handles=handles, loc='best')
plt.show()