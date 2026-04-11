import pandas as pd
from src.data_preparation import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.deployment import BatchPredictor

class ChurnPipeline:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()

    def run_full_pipeline(self):
        print("=== STARTING ML PIPELINE ===\n")
        
        df_clean = self.preprocessor.load_enrich_and_clean(self.raw_data_path)
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_train_test(df_clean)
        
        model = self.trainer.train(X_train, y_train)
        self.trainer.save_model()
        
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)
        
        predictor = BatchPredictor(model)
        
        # Use full dataset for batch prediction, exactly like the notebook simulation
        X = df_clean.drop(columns=[self.preprocessor.target_col])
        leads = predictor.generate_retention_leads(X, df_clean)
        
        print(f"\nPipeline Complete!")
        
        return leads

if __name__ == "__main__":
    RAW_DATA_FILE = "data/raw/Dataset.csv" 
    
    pipeline = ChurnPipeline(RAW_DATA_FILE)
    leads_dataframe = pipeline.run_full_pipeline()