import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    recall_score, precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)

def evaluate_model(
    model: Pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    save_outputs: bool = False,
    output_path: str = "outputs",
    figsize = (15, 7)):
    """
    Evaluate the model on the test set and print various metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_outputs:
        plt.savefig(f"{output_path}/plots/confusion_matrix.png")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label='ROC Curve', color='orange')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    # Add AUC score to the legend 
    plt.legend(loc='lower right', title=f'AUC: {roc_auc:.4f}')
    if save_outputs:
        plt.savefig(f"{output_path}/plots/roc_curve.png")

    # Plot precision and recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=figsize)
    plt.plot(thresholds, precision[:-1], label='Precision', color='red')
    plt.plot(thresholds, recall[:-1], label='Recall', color='blue')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.title('Precision and Recall by Threshold')
    plt.legend()
    if save_outputs:
        plt.savefig(f"{output_path}/plots/precision_recall.png")