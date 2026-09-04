"""
Customer Churn Risk Prediction Model

This module implements a Logistic Regression model to predict customer churn risk.
It includes data preprocessing, model training, evaluation, and visualization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnRiskPredictor:
    """
    A machine learning model for predicting customer churn risk.
    
    Attributes:
        model: LogisticRegression classifier
        scaler: StandardScaler for feature scaling
        feature_names: Names of features used in the model
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ChurnRiskPredictor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        return pd.read_csv(filepath)
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """
        Preprocess data: split and scale features.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model.
        
        Args:
            X_train: Scaled training features
            y_train: Training target variable
            
        Returns:
            self for method chaining
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.y_pred = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, self.y_pred),
            'precision': precision_score(y_test, self.y_pred, zero_division=0),
            'recall': recall_score(y_test, self.y_pred, zero_division=0),
            'f1': f1_score(y_test, self.y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, self.y_pred_proba)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance from model coefficients.
        
        Returns:
            DataFrame with features and their coefficients
        """
        if self.feature_names is None:
            raise ValueError("Feature names not set. Run preprocess_data first.")
        
        coefficients = self.model.coef_[0]
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot and display confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the figure (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the figure (optional)
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the figure (optional)
        """
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['abs_coefficient'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Absolute Coefficient Value')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """
    Main function to run the churn prediction pipeline.
    """
    print("Customer Churn Risk Prediction Model")
    print("=" * 50)
    print("\nNote: Update the data loading section with your CSV file path")
    print("Expected columns: features and target variable")
    

if __name__ == "__main__":
    main()
