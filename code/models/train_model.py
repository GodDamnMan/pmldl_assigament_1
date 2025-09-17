import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os
import joblib

def train_and_evaluate():
    # Load processed data
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Assuming the last column is the target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    # Start MLflow run
    mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(model.get_params())
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.pkl')
        
        print(f"Model trained with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_evaluate()