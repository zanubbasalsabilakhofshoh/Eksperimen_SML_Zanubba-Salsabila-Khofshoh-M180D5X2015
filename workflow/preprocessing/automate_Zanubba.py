
def load_dataset(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def encode_categorical(df):
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def scale_numeric(df, target_column):
    numeric_cols = df.drop(columns=[target_column]).select_dtypes(
        include=["int64", "float64"]
    ).columns

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

def preprocess_pipeline(path, target_column):
    df = load_dataset(path)
    df = clean_data(df)
    df = encode_categorical(df)
    df, scaler = scale_numeric(df, target_column)

    X_train, X_test, y_train, y_test = split_data(df, target_column)

    return df, X_train, X_test, y_train, y_test, scaler
from automate_Zanubba import preprocess_pipeline

PATH = "/content/drive/MyDrive/Eksperimen_SML_Zanubba/preprocessing/heart_disease_preprocessing/heart_disease_preprocessing.csv"
TARGET = "target"

df_clean, X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    PATH,
    TARGET
)

print("🚀 Preprocessing selesai!")
print("Shape data bersih:", df_clean.shape)
print("Train:", X_train.shape)
print("Test :", X_test.shape)
