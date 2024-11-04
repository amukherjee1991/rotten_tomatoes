import pandas as pd
import os
import zipfile

def unzip_folder(zip_path:str, extract_to:str):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to: {extract_to}")


def main():
    unzip_folder('datasets.zip', '.')

if __name__=="__main__":
    main()