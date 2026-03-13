import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns


# Ensure results directory exists
os.makedirs("results", exist_ok=True)


def plot_loss(losses):

    plt.figure(figsize=(8,5))

    plt.plot(losses, marker='o')

    plt.title("Training Loss vs Epoch")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig("results/loss_plot.png")

    plt.close()


def plot_accuracy(accuracies):

    plt.figure(figsize=(8,5))

    plt.plot(accuracies, marker='o')

    plt.title("Training Accuracy vs Epoch")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.savefig("results/accuracy_plot.png")

    plt.close()


def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.savefig("results/confusion_matrix.png")

    plt.close()


def plot_roc_curve(y_true, y_scores):

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    plt.plot([0,1], [0,1], linestyle="--")

    plt.title("ROC Curve")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")

    plt.savefig("results/roc_curve.png")

    plt.close()