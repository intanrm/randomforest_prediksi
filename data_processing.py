import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    data_encoded = pd.get_dummies(data, drop_first=True)
    feature_mapping = {col: data[col].unique().tolist() for col in data.columns}

    X = data_encoded.drop(columns=['Target'])
    y = data_encoded['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, feature_mapping
