import torch
from tqdm import tqdm
from config.config import DEVICE


def train_model(model, loader, optimizer, criterion, epochs):

    model.train()

    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):

        running_loss = 0
        correct = 0
        total = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in loop:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

    return train_losses, train_accuracies