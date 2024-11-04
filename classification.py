import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

os.makedirs("summary_stats", exist_ok=True)

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
    df["original_release_date"] = df["original_release_date"].fillna(placeholder_date)
    df["streaming_release_date"] = df["streaming_release_date"].fillna(placeholder_date)

    # Fill missing categorical values with 'Unknown'
    categorical_columns = [
        "content_rating",
        "genres",
        "directors",
        "authors",
        "actors",
        "production_company",
    ]
    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    # Encode content rating as numeric
    label_encoder = LabelEncoder()
    df["content_rating_encoded"] = label_encoder.fit_transform(df["content_rating"])

    # Fill missing numeric values with median
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
        df[column] = df[column].fillna(df[column].median())

    # Fill missing text columns with an empty string
    text_columns = ["movie_info", "critics_consensus"]
    for column in text_columns:
        df[column] = df[column].fillna("")

    # Drop any remaining rows with NaN in the target column (if necessary)
    df = df.dropna(subset=["tomatometer_status"])

    return df


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model.

    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Testing feature set
    - y_test: Testing labels

    Returns:
    - model: Trained Random Forest model
    - accuracy: Accuracy on the test set
    - report: Classification report on the test set
    """
    # Initialize the model
    model = RandomForestClassifier(random_state=42)
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Random Forest Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    return model, accuracy, report


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model.

    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Testing feature set
    - y_test: Testing labels

    Returns:
    - model: Trained Random Forest model
    - accuracy: Accuracy on the test set
    - report: Classification report on the test set
    """
    # Encode target labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    # y_test_encoded = label_encoder.transform(y_test)

    # Initialize and train the XGBoost model
    model = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="mlogloss"
    )
    model.fit(X_train, y_train_encoded)
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("XGBoost Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    return model, accuracy, report


def train_svc(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model.

    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Testing feature set
    - y_test: Testing labels

    Returns:
    - model: Trained Random Forest model
    - accuracy: Accuracy on the test set
    - report: Classification report on the test set
    """
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("SVC Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    return model, accuracy, report

def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    """
    Train and evaluate a Random Forest model.

    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Testing feature set
    - y_test: Testing labels

    Returns:
    - model: Trained Random Forest model
    - accuracy: Accuracy on the test set
    - report: Classification report on the test set
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("KNN Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    return model, accuracy, report

def train_mlp(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    """
    Train and evaluate a Random Forest model.

    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Testing feature set
    - y_test: Testing labels

    Returns:
    - model: Trained Random Forest model
    - accuracy: Accuracy on the test set
    - report: Classification report on the test set
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("MLP Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    return model, accuracy, report


def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Decision Tree model.

    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Testing feature set
    - y_test: Testing labels

    Returns:
    - model: Trained Decision Tree model
    - accuracy: Accuracy on the test set
    - report: Classification report on the test set
    """
    # Initialize the model
    model = DecisionTreeClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Decision Tree Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    return model, accuracy, report



def generate_summary_stats(df):
    # Save numeric summary statistics
    summary_stats = df.describe()
    summary_stats.to_csv("summary_stats/numeric_summary.csv")
    
    # Save categorical counts
    categorical_columns = ["content_rating", "genres", "directors", "authors", "actors", "production_company"]
    with open("summary_stats/categorical_counts.txt", "w") as f:
        for col in categorical_columns:
            f.write(f"{col}:\n")
            f.write(df[col].value_counts().to_string())
            f.write("\n\n")

def plot_distribution(df, column, title, filename):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x=column, palette="Set3")
    ax.bar_label(ax.containers[0])
    plt.title(title)
    plt.savefig(f"summary_stats/{filename}.png")
    plt.close()

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Rotten", "Fresh", "Certified-Fresh"], 
                yticklabels=["Rotten", "Fresh", "Certified-Fresh"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"summary_stats/{model_name}_confusion_matrix.png")
    plt.close()

def save_feature_importance(model, features):
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        indices = np.argsort(feature_importance)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), feature_importance[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.savefig("summary_stats/random_forest_feature_importance.png")
        plt.close()


def main():
    # Unzip and load data
    unzip_folder("datasets.zip", ".")
    reviews = read_csv("datasets/rotten_tomatoes_critic_reviews_50k.csv")
    movies = read_csv("datasets/rotten_tomatoes_movies.csv")
    merged_df = pd.merge(movies, reviews, on="rotten_tomatoes_link", how="left")
    merged_df = preprocess_data(merged_df)
    generate_summary_stats(movies)
    plot_distribution(movies, "content_rating", "Content Rating Distribution", "content_rating_distribution")
    plot_distribution(movies, "audience_status", "Audience Status Distribution", "audience_status_distribution")
    plot_distribution(movies, "tomatometer_status", "Tomatometer Status Distribution", "tomatometer_status_distribution")
    
    # Define target and features
    X = merged_df[
        ["runtime", "content_rating_encoded", "tomatometer_fresh_critics_count"]
    ]  # Add more features as needed
    y = merged_df["tomatometer_status"]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train and evaluate Random Forest
    model, accuracy, report = train_random_forest(X_train, y_train, X_test, y_test)
    model, accuracy, report = train_xgboost(X_train, y_train, X_test, y_test)
    model, accuracy, report = train_svc(X_train, y_train, X_test, y_test)
    model, accuracy, report = train_knn(X_train, y_train, X_test, y_test)
    model, accuracy, report = train_mlp(X_train, y_train, X_test, y_test)
    model, accuracy, report = train_decision_tree(X_train, y_train, X_test, y_test) 


if __name__ == "__main__":
    main()
