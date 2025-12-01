import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import traceback
import numpy as np

# Set style
sns.set(style="whitegrid")

def main():
    try:
        # Load data
        try:
            df = pd.read_csv("data (2).csv")
        except FileNotFoundError:
            print("Error: 'data (2).csv' not found.")
            return

        # Clean data
        df_cleaned = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        target_col = 'diagnosis'

        # 1. Class Distribution Plot
        print("Generating class distribution plot...")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_col, data=df_cleaned, palette='viridis', hue=target_col, legend=False)
        plt.title('Class Distribution (Benign vs Malignant)')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.savefig('class_distribution.png')
        plt.close()
        print("Generated class_distribution.png")

        # 2. Correlation Heatmap (Top 10 features correlated with target)
        print("Generating correlation heatmap...")
        # Encode target for correlation
        df_corr = df_cleaned.copy()
        df_corr[target_col] = df_corr[target_col].map({'B': 0, 'M': 1})
        
        # Calculate correlation with target
        correlations = df_corr.corrwith(df_corr[target_col])
        
        # Get top 10 features (absolute correlation)
        top_features = correlations.abs().nlargest(11).index.tolist()
        print(f"Top features: {top_features}")
        
        # Subset the correlation matrix
        corr_matrix = df_corr[top_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap (Top Features)')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        print("Generated correlation_heatmap.png")

        # 3. Model Training & Metrics
        print("Training model...")
        X = df_cleaned.drop(columns=[target_col])
        y = df_cleaned[target_col].map({'B': 0, 'M': 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Save metrics to a text file
        with open('metrics.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

        print("Generated metrics.txt")

        # Save Confusion Matrix Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Generated confusion_matrix.png")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
