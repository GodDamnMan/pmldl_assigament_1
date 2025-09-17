import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_process_data():
    # Load raw data
    raw_data_path = 'data/raw/data.csv'
    df = pd.read_csv(raw_data_path)
    
    # Data cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("Data processing completed successfully")

if __name__ == "__main__":
    load_and_process_data()