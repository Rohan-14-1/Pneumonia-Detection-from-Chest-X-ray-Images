import torch

DATASET_PATH = "dataset/chest_xray"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 0.001
MOMENTUM = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2
MODEL_SAVE_PATH = "checkpoints/model_weights.pth"