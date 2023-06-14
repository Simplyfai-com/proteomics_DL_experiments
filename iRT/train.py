import torch.nn as nn
import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from iRT.model import Transformer
from iRT.data import PRDataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    
model = Transformer(
        input_size=22, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

model.cuda()
    
batch_size = 1024
num_workers = 8

train_dataset = PRDataset(phase="train", data_path="/home/ubuntu/proto/data/x.npy", 
                          label_path="/home/ubuntu/proto/data/y.npy")

val_dataset = PRDataset(phase="validation", data_path="/home/ubuntu/proto/data/x_val.npy", 
                        label_path="/home/ubuntu/proto/data/y_val.npy")

train_dl = DataLoader(train_dataset, batch_size=batch_size, 
                    num_workers=num_workers, 
                    pin_memory=True, 
                    drop_last=True, 
                    shuffle=True)

val_dl = DataLoader(val_dataset, batch_size=batch_size*2, 
                    num_workers=num_workers, 
                    pin_memory=True,
                    drop_last=False, 
                    shuffle=False)


criterion = nn.L1Loss()
learning_rate = 1e-4
weight_decay = 1e-3
optimizer = optim.AdamW([
            {'params': model.parameters(),'lr': learning_rate, "weight_decay": weight_decay},
        ], betas=(0.9, 0.999), eps=1e-6)

exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1, verbose=False)


def train():

    model.train()
    optimizer.zero_grad()

    train_loss = 0.0 
    cnt = 0

    pbar = tqdm(train_dl)
    
    for data in pbar:

        cnt += 1
        
        sample = data
        data, label = sample['data'].cuda(), sample['label'].cuda()

        output = model(data)
        
        loss = criterion(output.float(), label.float()).float()
        train_loss += loss.item()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    train_epoch_loss = train_loss / cnt

    return train_epoch_loss


def eval():

    cnt = 0
    val_loss = 0.0
    pbar = tqdm(val_dl)
    model.eval()
    
    for data in pbar:

        cnt += 1
        sample = data
        data, label = sample['data'].cuda(), sample['label'].cuda()

        with torch.no_grad():
            output = model(data)
        
        loss = criterion(output.float(), label.float()).float()
        val_loss += loss.item()
    
    val_epoch_loss = val_loss / cnt

    return val_epoch_loss


stats = list()
best_train_loss = 1000.0
best_val_loss = 1000.0
epochs = 100
path = "/home/ubuntu/proto/models/"

for epoch in range(0,epochs):

    train_epoch_loss = train()
    val_epoch_loss = eval()

    exp_scheduler.step()
   
    if train_epoch_loss < best_train_loss:
        model_name = path + "train_ckpt.pth" 
        torch.save(model.state_dict(), model_name)
        best_train_loss = train_epoch_loss
    
    if val_epoch_loss < best_val_loss:
        model_name = path + "val_ckpt.pth" 
        torch.save(model.state_dict(), model_name)
        best_val_loss = val_epoch_loss

