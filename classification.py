import pandas as pd
import os
import zipfile

def unzip_folder(zip_path:str, extract_to:str):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to: {extract_to}")

def read_csv(file_path:str):
    df = pd.read_csv(file_path)
    return df



def main():
    unzip_folder('datasets.zip', '.')
    reviews = read_csv("datasets/rotten_tomatoes_critic_reviews_50k.csv")
    movies = read_csv("datasets/rotten_tomatoes_movies.csv")
    print(reviews)
    print(movies)

if __name__=="__main__":
    main()