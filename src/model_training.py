from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

class ModelTrainer:
    # Added max_depth=10 to match the best parameters from GridSearchCV
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        
        # We recreate the exact pipeline from the notebook
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth,          # <-- This prevents overfitting!
                random_state=random_state, 
                class_weight='balanced'
            ))
        ])

    def train(self, X_train, y_train):
        print("2. Trainer: Training the tuned Random Forest model...")
        self.model.fit(X_train, y_train)
        return self.model

    def save_model(self, filepath='churn_model.pkl'):
        print(f"2. Trainer: Saving trained model to {filepath}")
        joblib.dump(self.model, filepath)