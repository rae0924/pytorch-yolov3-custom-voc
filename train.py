from albumentations.augmentations import utils
from model import YOLOv3
from loss import YOLOv3Loss
from dataset import CustomVOCDataset
import torch
import torch.optim as optim
import config
import utils
import os
import numpy as np


# ds = CustomVOCDataset()
# net = YOLOv3(num_classes=ds.C)
# loss_fn = YOLOv3Loss()

# image, targets = ds[0]
# x = image.unsqueeze(0)
# y = net(x)

# total_loss = (
#     loss_fn(y[0], targets[0].unsqueeze(0), scaled_anchors[0])
#     + loss_fn(y[1], targets[1].unsqueeze(0), scaled_anchors[1])
#     + loss_fn(y[2], targets[2].unsqueeze(0), scaled_anchors[2])
# )
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_batch(x, y, model, opt, loss_fn, scaled_anchors):
    model.train()
    opt.zero_grad()
    out = model(x)
    loss = (
        loss_fn(out[0], y[0], scaled_anchors[0])
        + loss_fn(out[1], y[1], scaled_anchors[1])
        + loss_fn(out[2], y[2], scaled_anchors[2])
    )
    loss.backward()
    opt.step()
    return loss



def train_model(train_set, val_set, epochs=config.NUM_EPOCHS, path=config.SAVE_PATH):
    model = YOLOv3(num_classes=ds.C)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YOLOv3Loss()
    scaled_anchors = (torch.tensor(config.ANCHORS) * 
        torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    train_losses = []
    for epoch in range(epochs):
        train_epoch_losses = []
        for batch in train_set:
            images, targets = batch
            x, y = images, targets
            train_loss = train_batch(x, y, model, optimizer, loss_fn, scaled_anchors)
            print(train_loss)
            train_epoch_losses.append(train_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()
        train_losses.append(train_epoch_loss)
        utils.save_checkpoint(path, model, optimizer, epoch, history=[train_losses])



if __name__ == '__main__':
    ds = CustomVOCDataset()
    train_loader, val_loader = utils.train_val_split(ds, p=0.99, batch_size=config.BATCH_SIZE)
    train_model(train_loader, val_loader)
