import os
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sklearn.model_selection import train_test_split

from g2pk import G2p
from tqdm import tqdm

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def g2p_transform(text):
    g2p = G2p()
    text_g2p = g2p(text)
    
    if text_g2p != text :
        return text_g2p 
    else :
        return 0
    
def p2g_transform(text, tokenizer, model):
    inputs = tokenizer.encode(text,return_tensors="pt")
    output = model.generate(inputs)
    decoded_output = tokenizer.batch_decode(output[0], skip_special_tokens=True)
    
    text_p2g = ''.join(decoded_output)
    
    # 변환 과정중 발생하는 � 없애기
    text_p2g = text_p2g.replace('�', '')
    
    return text_p2g 

def main() :
    data = pd.read_csv('../data/train.csv')
    data = data.iloc[:5]
  
    # g2p로 filtering
    tqdm.pandas()
    data['text_g2p'] = data['text'].progress_apply(g2p_transform)
    filtered_data = data[data['text_g2p'] == 0]
    
    # g2p로 filtering한 데이터를 train_g2p.csv 저장
    filtered_data.to_csv('../data/train_g2p.csv')
        
    # g2p로 거른 데이터를 모아 다시 원본 문장으로 복구
    model_dir = "kfkas/t5-large-korean-P2G"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    
    tqdm.pandas()
    filtered_data['text_p2g'] = filtered_data['text'].progress_apply(p2g_transform, tokenizer=tokenizer, model=model)
    
    # 복구한 문장을 넣은 data를 train_p2g.csv로 저장
    filtered_data.to_csv('../data/train_p2g.csv')


if __name__ == "__main__":
    main()