import pandas as pd
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier


def unzip_folder(zip_path: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to: {extract_to}")


def read_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    # Convert dates to datetime and fill missing values with a placeholder date
    df["original_release_date"] = pd.to_datetime(
        df["original_release_date"], errors="coerce"
    )
    df["streaming_release_date"] = pd.to_datetime(
        df["streaming_release_date"], errors="coerce"
    )
    placeholder_date = pd.to_datetime("1900-01-01")
    df["original_release_date"].fillna(placeholder_date, inplace=True)
    df["streaming_release_date"].fillna(placeholder_date, inplace=True)

    categorical_columns = [
        "content_rating",
        "genres",
        "directors",
        "authors",
        "actors",
        "production_company",
    ]
    for column in categorical_columns:
        df[column].fillna("Unknown", inplace=True)

    label_encoder = LabelEncoder()
    df["content_rating_encoded"] = label_encoder.fit_transform(df["content_rating"])
    numeric_columns = [
        "runtime",
        "tomatometer_rating",
        "tomatometer_count",
        "audience_rating",
        "audience_count",
        "tomatometer_top_critics_count",
        "tomatometer_fresh_critics_count",
        "tomatometer_rotten_critics_count",
    ]
    for column in numeric_columns:
        df[column].fillna(df[column].median(), inplace=True)
    text_columns = ["movie_info", "critics_consensus"]
    for column in text_columns:
        df[column].fillna("", inplace=True)

    df = df.dropna(subset=["tomatometer_status"])
    return df


def main():
    # Unzip and load data
    unzip_folder("datasets.zip", ".")
    reviews = read_csv("datasets/rotten_tomatoes_critic_reviews_50k.csv")
    movies = read_csv("datasets/rotten_tomatoes_movies.csv")

    # Merge datasets
    merged_df = pd.merge(movies, reviews, on="rotten_tomatoes_link", how="left")
    merged_df = preprocess_data(merged_df)

    # Define target and features
    X = merged_df[
        ["runtime", "content_rating_encoded", "tomatometer_fresh_critics_count"]
    ]  # Add more features as needed
    y = merged_df["tomatometer_status"]
    print(X)
    print(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check for NaN values in X_train and y_train
    print("Checking for NaN values in X_train:")
    print(X_train.isna().sum())
    print("Checking for NaN values in y_train:")
    print(y_train.isna().sum())

    # # Initialize model
    model = RandomForestClassifier()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
