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


from model import *
from train import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current divice is', device)
print('current torch version is', torch.__version__)


data_clean8 = pd.read_csv('/opt/ml/data_clean8_saved.csv')
data_clean8['target'] = 0
data_dirty8 = pd.read_csv('/opt/ml/data_dirty8_saved.csv')
data_dirty8['target'] = 1
dataset_train = pd.concat([data_clean8, data_dirty8], axis=0)
data_clean2 = pd.read_csv('/opt/ml/data_clean2_saved.csv')
data_clean2['target'] = 0
data_dirty2 = pd.read_csv('/opt/ml/data_dirty2_saved.csv')
data_dirty2['target'] = 1
dataset_val = pd.concat([data_clean2, data_dirty2], axis=0)


data_train = denoiseDataset(dataset_train, tokenizer)
data_val = denoiseDataset(dataset_val, tokenizer)
train_dataloader = DataLoader(data_train, batch_size=128, shuffle=True)
val_dataloader = DataLoader(data_val, batch_size=128)


model = Transformer(num_tokens=399, dim_model=256, num_heads=4,
                    num_encoder_layers=8).to(device)
opt = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
loss_fn = nn.BCEWithLogitsLoss()


epochs_used=1
train_loss_list, validation_loss_list, mixed_list = fit(model, opt, 
                                            loss_fn, train_dataloader,
                                            val_dataloader, epochs_used, device)


plt.plot(train_loss_list, label = "Train loss")
plt.plot(validation_loss_list, label = "Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()
plt.plot(mixed_list, label = 'mixed_list')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.legend()
plt.show()
print(train_loss_list)
print(validation_loss_list)
print(mixed_list)


examples = ['나는 집에 가고 싶어', '나는 지베 가고 시퍼',
            '이 물건의 값이 궁금해', '이 물거네 갑씨 궁금해',
            '모든 것 쏟아낸 워싱턴 셔저 5차전 등판 못 할 듯종합',
            '모든 걷 쏘다내 눠싱턴 셔저 오차전 등판 모 탈 뜯쫑합',
            'KIST 가장 혁신적인 25개 연구기관 중 6위에 선정',
            '키스트 가장 혁씬저긴 스물다섣깨 연구기관 중 유귀에 선정']
for idx, example in enumerate(examples):
    result = predict(model, example, device)
    print(example)
    print(result)
    print()


torch.save(model.state_dict(), '/opt/ml/model_denoising_state_dict.pt' )