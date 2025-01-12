import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('/mnt/data/data_kb.csv')

# 1. Struktur Dataset
print("Struktur Dataset:")
print(data.info())
print("\nLima Baris Pertama Dataset:")
print(data.head())

# 2. Statistik Deskriptif
print("\nStatistik Deskriptif:")
print(data.describe(include='all'))

# 3. Keseimbangan Kelas Target
print("\nDistribusi Kelas Target:")
print(data['penggunaan_kb'].value_counts(normalize=True))
sns.countplot(x='penggunaan_kb', data=data)
plt.title("Distribusi Kelas Target")
plt.show()

# 4. Korelasi Antar Fitur
# Hanya untuk fitur numerik
print("\nKorelasi Antar Fitur Numerik:")
correlation = data.corr()
print(correlation)
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# 5. Missing Values
print("\nJumlah Missing Values di Setiap Kolom:")
print(data.isnull().sum())

# 6. Distribusi Data Kategorikal
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nDistribusi Data Kategorikal:")
for col in categorical_columns:
    print(f"\nKolom: {col}")
    print(data[col].value_counts())
    sns.countplot(x=col, data=data)
    plt.title(f"Distribusi {col}")
    plt.xticks(rotation=45)
    plt.show()

# 7. Deteksi Outlier (Boxplot untuk fitur numerik)
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot {col}")
    plt.show()
