# test_loader.py

from src.data_loader import load_data, inspect_data

# Point to your dataset
file_path = 'data/creditcard.csv'

# Load & inspect
df = load_data(file_path)
if df is not None:
    inspect_data(df)
# Ensure the data_loader module is in the src directory