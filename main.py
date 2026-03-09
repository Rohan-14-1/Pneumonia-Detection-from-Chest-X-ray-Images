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

    train_loader, test_loader = load_datasets(DATASET_PATH)

    model = CNNModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = LipschitzMomentum(
        model.parameters(),
        lr=LEARNING_RATE,
        beta=MOMENTUM
    )

    losses = train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        EPOCHS
    )

    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

    print("\nResults")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    plot_loss(losses)
    plot_accuracy([accuracy])

    torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()