import matplotlib.pyplot as plt
import os

def plot_loss(train_losses):

    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/loss_plot.png")
    plt.close()


def plot_accuracy(acc):

    plt.figure()
    plt.plot(acc)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("results/accuracy_plot.png")
    plt.close()