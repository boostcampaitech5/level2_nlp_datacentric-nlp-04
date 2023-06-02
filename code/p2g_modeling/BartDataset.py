import torch
from torch.utils.data import Dataset
import random
import string

def random_str():
    rand_str = ''
    for i in range(3):
        rand_str += str(random.choice(string.ascii_uppercase + string.digits))
    return rand_str

class TranslationDataset(Dataset) :
    def __init__(self, data, tokenizer):
        input_texts = data['g2p']
        targets = data['text']
        self.inputs = []
        self.labels = []
        
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt',max_length=256)
            tokenized_target = tokenizer(label, padding='max_length', truncation=True, return_tensors='pt',max_length=256)
            self.inputs.append(tokenized_input)
            self.labels.append(tokenized_target['input_ids'])
            
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)