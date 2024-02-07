import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.utils.multiclass import unique_labels

# Function to train models
def train_models(X, y, model, model_name):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    model_fitted = model.fit(train_x, train_y)
    predictions = model_fitted.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions, average='weighted')
    recall = recall_score(test_y, predictions, average='weighted')
    f1 = f1_score(test_y, predictions, average='weighted')
    
    # Handle SVM's lack of predict_proba
    if hasattr(model, 'predict_proba'):
        test_x = np.ascontiguousarray(test_x)
        y_pred_proba = model_fitted.predict_proba(test_x)
        # Plot ROC curve
        plot_roc_curve(test_y, y_pred_proba, model_name)

    # Plot confusion matrix
    plot_confusion_matrix(test_y, predictions, model_name)

    return {"F1-Score": f1, "Precision": precision, "Recall": recall, "Accuracy": accuracy}

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_proba, model_name):
    plt.figure(figsize=(10, 8))
    for i in range(y_pred_proba.shape[1]):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for class {i}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curves - {model_name}')
    
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    classes = unique_labels(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Load dataset and preprocess
train_dataset = pd.read_csv("Training.csv")
train_dataset.pop('Unnamed: 133')
target = train_dataset['prognosis']
train_dataset.pop('prognosis')

# Encode target variable
encoder = LabelEncoder()
target = encoder.fit_transform(target)

# Shuffle dataset
train_dataset = shuffle(train_dataset, random_state=42)

# Define classification models
classification_models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1, random_state=666),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),  # Enable probability estimates for SVM
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier()
}

# Train and evaluate models
for model_name, model in classification_models.items():
    print("Model: " + model_name)
    train_models(train_dataset, target, model, model_name)