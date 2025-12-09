import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def encode_categorical(df):
    categorical_cols = df.select_dtypes(include="object").columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def scale_numeric(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def preprocess_pipeline(path, target_column):
    df = load_dataset(path)
    df_clean = clean_data(df)
    df_clean, encoders = encode_categorical(df_clean)
    df_clean, scaler = scale_numeric(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_clean, target_column)
    return df_clean, X_train, X_test, y_train, y_test, encoders, scaler

from automate_Nama_siswa import preprocess_pipeline

PATH = "/content/data_siswa.csv"
TARGET = "target"

df_clean, X_train, X_test, y_train, y_test, encoders, scaler = preprocess_pipeline(
    PATH,
    TARGET
)

print("🚀 Preprocessing selesai!")
print("Shape data bersih:", df_clean.shape)
print("Train:", X_train.shape)
print("Test :", X_test.shape)
