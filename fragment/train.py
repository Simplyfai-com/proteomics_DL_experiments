import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from functools import partial
from fragment.model import FragmentModel, masked_spectral_Loss, LayerNorm
from fragment.data import PRDataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

scaler = GradScaler()

    
model = FragmentModel(
        input_size=22, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(LayerNorm, eps=1e-6))

state_dict = torch.load("/home/ubuntu/proto/models/iteration06/val_ckpt.pth")
model.pep_encoder.load_state_dict(state_dict, strict=True)

model.cuda()

    
batch_size = 1024
num_workers = 8

train_dataset = PRDataset(phase="train",
                          data_path="/home/ubuntu/proto/data/prosit_fragmentation/prosit_fragmentation_train.lmdb")

val_dataset = PRDataset(phase="validation", 
                        data_path="/home/ubuntu/proto/data/prosit_fragmentation/prosit_fragmentation_test.lmdb")

train_dl = DataLoader(train_dataset, batch_size=batch_size, 
                    num_workers=num_workers, 
                    pin_memory=True, 
                    drop_last=True, 
                    shuffle=True)

val_dl = DataLoader(val_dataset, batch_size=batch_size, 
                    num_workers=num_workers, 
                    pin_memory=True,
                    drop_last=False, 
                    shuffle=False)


criterion = masked_spectral_Loss()

learning_rate = 1e-4
weight_decay = 1e-5

optimizer = optim.AdamW([
            {'params': model.parameters(),'lr': learning_rate, "weight_decay": weight_decay},
        ], betas=(0.9, 0.999), eps=1e-9)


exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=False)


def train(epoch_count):

    model.train()
    epoch_count += 1

    if epoch_count == 1:
        model.freeze_backbone()
        freeze = True

    train_loss = 0.0 
    cnt = 0
    optimizer.zero_grad()
    pbar = tqdm(train_dl)
    
    for data in pbar:

        cnt += 1

        if freeze and cnt > 4500:
            freeze = False    
            model.unfreeze_backbone()
        
        sample = data
        precursor_charge, collision_energy, peptide_sequence, label = sample['precursor_charge'].cuda(), sample['collision_energy'].cuda(), sample['peptide_sequence'].cuda(), sample['label'].cuda()

        with autocast():
            output = model(peptide_sequence, precursor_charge, collision_energy)
        
        loss = criterion(output, label)
        train_loss += loss.item()
        
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
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
        precursor_charge, collision_energy, peptide_sequence, label = sample['precursor_charge'].cuda(), sample['collision_energy'].cuda(), sample['peptide_sequence'].cuda(), sample['label'].cuda()

        with autocast():
            with torch.no_grad():
                output = model(peptide_sequence, precursor_charge, collision_energy)
        
        loss = criterion(output, label)
        val_loss += loss.item()

    
    val_epoch_loss = val_loss / cnt

    return val_epoch_loss


stats = list()
best_train_loss = 1000.0
best_val_loss = 1000.0
epochs = 100
path = "/home/ubuntu/proto/models/"

for epoch in range(0,epochs):

    epoch_count = epoch
    
    train_epoch_loss = train(epoch_count)
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

