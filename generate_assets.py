import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
import os
import traceback
import numpy as np

# Set style for better-looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def main():
    try:
        print("="*60)
        print("Generating Assets for Breast Cancer Classification Project")
        print("="*60)
        
        # Load data
        try:
            df = pd.read_csv("data (2).csv")
            print(f"\n✓ Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        except FileNotFoundError:
            print("Error: 'data (2).csv' not found.")
            return

        # Clean data
        df_cleaned = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        target_col = 'diagnosis'
        print(f"✓ Data cleaned: {df_cleaned.shape[1]} features after removing ID columns")

        # 1. Class Distribution Plot
        print("\n[1/4] Generating class distribution plot...")
        plt.figure(figsize=(10, 7))
        counts = df_cleaned[target_col].value_counts()
        colors = ['#2ecc71', '#e74c3c']  # Green for Benign, Red for Malignant
        
        ax = sns.countplot(x=target_col, data=df_cleaned, palette=colors, hue=target_col, legend=False)
        plt.title('Class Distribution: Benign vs Malignant', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Diagnosis', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Cases', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (label, count) in enumerate(counts.items()):
            percentage = (count / len(df_cleaned)) * 100
            ax.text(i, count + 5, f'{count}\n({percentage:.1f}%)', 
                   ha='center', fontsize=12, fontweight='bold')
        
        # Add legend with full names
        ax.set_xticklabels(['Benign (B)', 'Malignant (M)'], fontsize=12)
        plt.tight_layout()
        plt.savefig('class_distribution.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Generated: class_distribution.png")

        # 2. Correlation Heatmap (Top 10 features correlated with target)
        print("\n[2/4] Generating correlation heatmap...")
        # Encode target for correlation
        df_corr = df_cleaned.copy()
        df_corr[target_col] = df_corr[target_col].map({'B': 0, 'M': 1})
        
        # Calculate correlation with target
        correlations = df_corr.corrwith(df_corr[target_col])
        
        # Get top 10 features (absolute correlation)
        top_features = correlations.abs().nlargest(11).index.tolist()
        print(f"  • Top correlated features: {', '.join(top_features[:5])}")
        
        # Subset the correlation matrix
        corr_matrix = df_corr[top_features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', fmt='.2f', 
                   linewidths=0.5, square=True, mask=mask, 
                   cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap (Top 10 Features)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Generated: correlation_heatmap.png")

        # 3. Model Training & Metrics
        print("\n[3/4] Training Random Forest model...")
        X = df_cleaned.drop(columns=[target_col])
        y = df_cleaned[target_col].map({'B': 0, 'M': 1})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  • Training samples: {len(X_train)}")
        print(f"  • Testing samples: {len(X_test)}")

        clf = RandomForestClassifier(random_state=42, n_estimators=100)
        clf.fit(X_train, y_train)
        print("  ✓ Model training complete")

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Save comprehensive metrics to a text file
        with open('metrics.txt', 'w') as f:
            f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
            f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
            f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"                Predicted\n")
            f.write(f"              Benign  Malignant\n")
            f.write(f"Actual Benign    {cm[0][0]:3d}      {cm[0][1]:3d}\n")
            f.write(f"     Malignant   {cm[1][0]:3d}      {cm[1][1]:3d}\n")

        print(f"  ✓ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("  ✓ Generated: metrics.txt")

        # Save Confusion Matrix Plot with enhanced visualization
        print("\n[4/4] Generating confusion matrix visualization...")
        plt.figure(figsize=(10, 8))
        
        # Create annotations with both count and percentage
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                   xticklabels=['Benign (0)', 'Malignant (1)'], 
                   yticklabels=['Benign (0)', 'Malignant (1)'],
                   cbar_kws={"shrink": 0.8}, linewidths=2, linecolor='white')
        
        plt.title('Confusion Matrix - Model Predictions', fontsize=18, fontweight='bold', pad=20)
        plt.ylabel('Actual Diagnosis', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Diagnosis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Generated: confusion_matrix.png")

        # Save the trained model
        joblib.dump(clf, 'breast_cancer_model.joblib')
        print("\n✓ Model saved: breast_cancer_model.joblib")

        print("\n" + "="*60)
        print("Asset Generation Complete!")
        print("="*60)
        print("\nGenerated files:")
        print("  • class_distribution.png")
        print("  • correlation_heatmap.png")
        print("  • confusion_matrix.png")
        print("  • metrics.txt")
        print("  • breast_cancer_model.joblib")
        print("\nYou can now run: python create_ppt.py")
        print("="*60)

    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
