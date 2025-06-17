import pandas as pd

def load_data(filepath):
    """Loads the credit card dataset from the given file path."""
    try:
        df = pd.read_csv(filepath)
        print("✅ Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ File not found at {filepath}")
        return None

def inspect_data(df):
    """Prints basic information about the dataset."""
    print("\n📊 Dataset Info:")
    print(df.info())
    print("\n🔍 Null Values:\n", df.isnull().sum())
    print("\n🔢 Descriptive Stats:\n", df.describe())
    print("\n🪪 Class Distribution:\n", df['Class'].value_counts())
