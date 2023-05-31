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

from denoise_for_train.model import *
from denoise_for_train.train import *


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(num_tokens=399, dim_model=256, num_heads=4,
                    num_encoder_layers=8).to(device)


## Change model path
model_state_dict_path = "/opt/ml/level2_nlp_datacentric-nlp-04/code/denoise_for_train/model_denoising_state_dict_256_5e6_09.pt"
noise_csv_path = '/opt/ml/data/train.csv'
to_total_path = '/opt/ml/train_after_total_09.csv'


model_state_dict = torch.load(model_state_dict_path, map_location=device)
model.load_state_dict(model_state_dict)
df = pd.read_csv(noise_csv_path)


logits = []
for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='df.iterrows()'):
    #4min 30sec
    single_sentence = row['text']
    pred = predict(model, single_sentence, device)
    logit, result = pred[0].item(), pred[1]
    logits.append(logit)
print('The threshold for 10%, 15%, 20%, 25%, are')
logits_temp = sorted(logits)
print(logits_temp[int(-0.10*(len(logits_temp)))],
      logits_temp[int(-0.15*(len(logits_temp)))],
      logits_temp[int(-0.20*(len(logits_temp)))],
      logits_temp[int(-0.25*(len(logits_temp)))]
      )
plt.hist(logits, bins=70, edgecolor='black')
plt.xlabel('Logits')
plt.ylabel('Frequency')
plt.title('Histogram of Logits')
plt.show()

df['is_noise'] = logits
df.to_csv(to_total_path, index=False)


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