import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from network import UNet, UNetSiamese

LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
LOSS_FN = nn.CrossEntropyLoss()
# model = UNet().to(DEVICE)
# model = UNetSiamese().to(DEVICE)
# optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
BATCH_SIZE = 8