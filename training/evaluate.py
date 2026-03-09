import torch
from config.config import DEVICE
from utils.metrics import compute_metrics

def evaluate_model(model, loader):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

    return accuracy, precision, recall, f1