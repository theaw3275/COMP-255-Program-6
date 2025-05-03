# main

# imports
import pandas as pd
import numpy as np

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