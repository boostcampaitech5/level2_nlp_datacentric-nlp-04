import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support




def tokenizer(texts: str, max_len=150):
    '''
    Tokenize Hangeul consonant, Hangeul vowel, Roman alphabet and number.
    pad_token = 10
    cls_token = 11

    Returns:
    token: list whose len is max_len consists of sparse int 0~327
    '''

    token = []
    for char in texts:
        if char.isdecimal():
            token.append(int(char)) #0-9
            token.append(int(char))
            token.append(int(char))
        elif ord(char)>=ord('가') and ord(char)<=ord('힣'):
            temp = ord(char)
            temp = temp - 44032
            onset = temp//(21*28) #가, 까
            temp = temp%(21*28)
            nucl = temp//28 #가, 개
            temp = temp%28
            coda = temp//1 #가, 각
            token.append(onset+100)
            token.append(nucl+200)
            token.append(coda+300)
        elif char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            char = char.lower()
            token.append(ord(char)-23) #a-z=97-122, A-Z=65-90
            token.append(ord(char)-23)
            token.append(ord(char)-23)
        else:
            pass
    token = [11] + token + (max_len-len(token))*[10]
    token = token[:max_len]
    return token


class denoiseDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        transform = tokenizer
        texts = dataset['input_text'].tolist()
        targets = dataset['target'].tolist()
        self.sentences = [transform(i) for i in texts]
        self.labels = [np.int32(i) for i in targets]

    def __getitem__(self, i):
        return (self.sentences[i], self.labels[i])

    def __len__(self):
        return (len(self.labels))
    
    
def train_loop(model, opt, loss_fn, dataloader, device):
    model.train()
    total_loss = 0

    y_true, y_pred = [], []
    for batch in tqdm(dataloader, desc='train loop'):
        X, y = batch[0], batch[1]
        # print(len(X), len(X[0]), (X[0][0])) #150 64 tensor(11)
        X = torch.stack(X).to(device) 
        # print(X.shape) #torch.Size([150, 64])
        y = y.view(-1, 1).float().to(device)
        # print(y.shape) #torch.Size([64, 1])

        pred = model(X)
        loss = loss_fn(pred, y)
        # print(pred.shape) #torch.Size([64, 1])
        opt.zero_grad()
        loss.backward()
        opt.step()    

        total_loss += loss.detach().item()
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend((pred>0).float().detach().cpu().tolist())
    c_m = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(c_m)
    prfs = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    print(f'(label=0|true=0)={prfs[1][0]}, (label=1|true=1)={prfs[1][1]}')
    return round(total_loss / len(dataloader), 3)


def validation_loop(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='val loop'):
            X, y = batch[0], batch[1]
            X = torch.stack(X).to(device)
            y = y.view(-1, 1).float().to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.detach().item()
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend((pred>0).float().detach().cpu().tolist())
    c_m = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(c_m)
    prfs = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    print(f'(label=0|true=0)={prfs[1][0]}, (label=1|true=1)={prfs[1][1]}')
    return round(total_loss / len(dataloader), 3), round((prfs[1][0]+prfs[1][1])/2, 3)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device):  
    train_loss_list, validation_loss_list, mixed_list = [], [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device)
        train_loss_list += [train_loss]

        validation_loss, mixed_value = validation_loop(model, loss_fn, val_dataloader, device)
        validation_loss_list += [validation_loss]
        mixed_list += [mixed_value]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"mixed_value: {mixed_value:.4f}")
        print()
    return train_loss_list, validation_loss_list, mixed_list


def predict(model, single_sentence: str, device):
    '''
    Returns:
    logit:float. If logit > 0, it is noise
    is_noise:bool. If True, it is noise
    '''
    with torch.no_grad():
        model.eval()
        X = single_sentence
        X = tokenizer(X)
        X = torch.tensor(X).to(device).view(-1, 1)
        pred = model(X).detach().cpu()
        logit, is_noise = pred, (pred>0)
    return logit, is_noise