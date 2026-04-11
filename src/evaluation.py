from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        print("3. Evaluator: Generating model metrics...")
        predictions = self.model.predict(X_test)
        
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        print("\n--- Model Performance ---")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        print("-------------------------\n")
        
        return {"precision": precision, "recall": recall, "f1": f1}