from model import YOLOv3
from loss import YOLOv3Loss
from dataset import CustomVOCDataset
import torch
import torch.optim as optim
import config
import utils
import numpy as np


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


def train_model(train_set, val_set, epochs=config.NUM_EPOCHS, save_path=config.SAVE_PATH, cp_path=config.CP_PATH):
    model = YOLOv3(num_classes=ds.C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    model, optimizer, epochs_done, history = utils.load_checkpoint(cp_path, model, optimizer)
    loss_fn = YOLOv3Loss()
    scaled_anchors = (torch.tensor(config.ANCHORS) * 
        torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(device)

    train_losses = []
    if len(history)!=0:
        train_losses = history[0]

    for epoch in range(epochs_done, epochs):
        train_epoch_losses = []
        for batch in train_set:
            images, targets = batch
            x = images.to(device)
            y = [t.to(device) for t in targets]
            train_loss = train_batch(x, y, model, optimizer, loss_fn, scaled_anchors)
            train_epoch_losses.append(train_loss.detach().cpu())
        train_epoch_loss = np.array(train_epoch_losses).mean()
        print(f'epoch: {epoch+1}, mean_loss: {train_epoch_loss}')
        train_losses.append(train_epoch_loss)
        utils.save_checkpoint(save_path, model, optimizer, epoch+1, history=[train_losses])



if __name__ == '__main__':
    ds = CustomVOCDataset()
    train_loader, val_loader = utils.train_val_split(ds, p=0, batch_size=config.BATCH_SIZE)
    train_model(train_loader, val_loader)
