import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Clean column names
    data.columns = data.columns.str.lower().str.replace(' ', '_')

    # Categorical columns to encode
    categorical_columns = ['jenis_kelamin', 'pendidikan', 'status_ekonomi', 'pekerjaan', 'sumber_informasi', 'penggunaan_kb']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Normalize numerical columns
    scaler = MinMaxScaler()
    data[['usia', 'jumlah_anak']] = scaler.fit_transform(data[['usia', 'jumlah_anak']])

    # Prepare features and target
    feature_columns = ['usia', 'jumlah_anak', 'jenis_kelamin', 'pendidikan', 'status_ekonomi', 'pekerjaan', 'sumber_informasi']
    X = data[feature_columns]
    y = data['penggunaan_kb']

    # Visualize class distribution before SMOTE
    print("Distribusi kelas sebelum SMOTE:")
    print(y.value_counts())
    sns.countplot(x=y)
    plt.title("Distribusi Kelas Sebelum SMOTE")
    plt.show()

    # Handle class imbalance with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, y = smote.fit_resample(X, y)

    # Visualize class distribution after SMOTE
    print("Distribusi kelas setelah SMOTE:")
    print(pd.Series(y).value_counts())
    sns.countplot(x=y)
    plt.title("Distribusi Kelas Setelah SMOTE")
    plt.show()

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders, scaler
