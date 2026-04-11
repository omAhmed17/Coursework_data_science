import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_enrichment import DataScraper

class DataPreprocessor:
    def __init__(self, target_col='Churn Value'):
        self.target_col = target_col
        self.drop_cols = ['City', 'Zip Code', 'Customer ID', 'Churn Category', 'Churn Label', 'Churn Reason', 'Churn']
        
        # Instantiate the scraper so the Preprocessor owns it
        self.scraper = DataScraper() 

    # Make sure this is named load_enrich_and_clean!
    def load_enrich_and_clean(self, filepath):
        print("1. Preprocessor: Loading raw data from file...")
        df_raw = pd.read_csv(filepath)
        
        print("2. Preprocessor: Asking Scraper to enrich the data...")
        df_enriched = self.scraper.enrich_data(df_raw)
        
        print("3. Preprocessor: Cleaning the enriched data...")
        df_enriched['Offer'] = df_enriched['Offer'].fillna(df_enriched['Offer'].mode()[0])
        df_enriched['Internet Type'] = df_enriched['Internet Type'].fillna(df_enriched['Internet Type'].mode()[0])
        df_enriched['Customer Satisfaction'] = df_enriched['Customer Satisfaction'].fillna(df_enriched['Customer Satisfaction'].median())
        
        cols_to_drop = [col for col in self.drop_cols if col in df_enriched.columns]
        df_clean = df_enriched.drop(columns=cols_to_drop, errors='ignore')
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        
        return df_encoded

    def prepare_train_test(self, df_encoded):
        print("4. Preprocessor: Splitting data into Train and Test sets...")
        y = df_encoded[self.target_col]
        X = df_encoded.drop(columns=[self.target_col])
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)