# main

# imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

"""Clustering the documents using K-Means"""
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
kmeans = KMeans(n_clusters=n_clusters)
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
#plt.show()

"""SVM"""

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y
)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# put y in the right format
y_train = np.array(y_train).flatten()

# Create and train the SVM classifier
clf = SVC(kernel='linear', C=1, class_weight='balanced')
clf.fit(X_train, y_train)

unique_train, counts_train = np.unique(y_train, return_counts=True)
print("Training set distribution:")
for label, count in zip(unique_train, counts_train):
    print(f"Class {label}: {count}")

# Predict on the test set
y_pred = clf.predict(X_test)

# Compute accuracy (rounded to 2 decimal places)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Classification Accuracy on test set: {accuracy:.2f}")

# Print a full classification report (precision, recall, f1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=uniqueClasses))

# Optionally, create and display a confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=uniqueClasses)
cm_df = pd.DataFrame(cm, index=uniqueClasses, columns=uniqueClasses)
print("Confusion Matrix (Rows: Actual, Columns: Predicted):")
print(cm_df)