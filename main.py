import torch
import torch.nn as nn

from config.config import *
from data.dataset_loader import load_datasets
from models.cnn_model import CNNModel
from optimizers.lipschitz_momentum import LipschitzMomentum
from training.train import train_model
from training.evaluate import evaluate_model
from utils.plot_results import plot_loss, plot_accuracy


def main():

    print("Loading dataset...")

    train_loader, test_loader = load_datasets(DATASET_PATH)

    print("Initializing model...")

    model = CNNModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = LipschitzMomentum(
        model.parameters(),
        lr=LEARNING_RATE,
        beta=MOMENTUM
    )

    print("Starting training...")

    losses, accuracies = train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        EPOCHS
    )

    print("Evaluating model...")

    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

    print("\nFinal Evaluation Results")
    # print("----------------------------")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")

    print("\nSaving graphs...")

    plot_loss(losses)
    plot_accuracy(accuracies)

    print("Graphs saved in results/ folder")

    print("Saving model...")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Model saved successfully!")


if __name__ == "__main__":
    main()