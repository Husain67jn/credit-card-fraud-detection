import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # Drop the 'Time' column
    df = df.drop(columns=['Time'])

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Scale the 'Amount' column only
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])

    # Train-test split before SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to training data only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test
