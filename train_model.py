import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_activity_model(csv_path='training_master.csv'):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run feature extraction first.")
        return
        
    df = pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Dataset loaded. Total shape: {df.shape}")
    
    # Check if necessary columns exist
    required_cols = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                     'left_hip', 'right_hip', 'left_knee', 'right_knee', 'activity']
    
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in {csv_path}.")
            return

    # Extract Features and Labels
    # Selecting the 8 joint angle columns for input
    X = df[['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
            'left_hip', 'right_hip', 'left_knee', 'right_knee']]
    y = df['activity']

    # Train-test split (80-20)
    print("Splitting data into 80% training and 20% validation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training RandomForest Classifier...")
    # Initialize Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf_clf.fit(X_train, y_train)
    
    # Validation predictions
    print("Evaluating model...")
    y_pred = rf_clf.predict(X_test)
    
    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Performance ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=rf_clf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_clf.classes_, yticklabels=rf_clf.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Activity Recognition')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Save the model
    model_filename = 'activity_model.pkl'
    joblib.dump(rf_clf, model_filename)
    print(f"Trained model saved as '{model_filename}'")
    
if __name__ == "__main__":
    train_activity_model()
