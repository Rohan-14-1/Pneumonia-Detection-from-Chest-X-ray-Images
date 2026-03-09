import torch
from tqdm import tqdm
from config.config import DEVICE

def train_model(model, loader, optimizer, criterion, epochs):

    model.train()

    losses = []

    for epoch in range(epochs):

        running_loss = 0

        loop = tqdm(loader)

        for images, labels in loop:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(loader)

        losses.append(epoch_loss)

    return losses