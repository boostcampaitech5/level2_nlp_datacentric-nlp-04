import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer,AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from BartDataset import TranslationDataset

def main():
    model_name = 'cosmoquester/bart-ko-base' # Example model
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../../data')
    
    data = pd.read_csv(os.path.join(DATA_DIR, 'sinmoon_news_g2p.csv'))
    
    dataset_train, dataset_valid = train_test_split(data, test_size=0.1, random_state=42)
   
    train_dataset = TranslationDataset(data,tokenizer)  # Your training dataset
    eval_dataset = TranslationDataset(dataset_valid,tokenizer) # Your evaluation dataset
    
    data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./fine-tuned-model",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        weight_decay=0.01,
        logging_steps = 50,
        dataloader_num_workers=0
    )
    
    # Define the fine-tuning trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator = data_collator,
    )
    
    # Fine-tune the model
    trainer.train()
    
    model_save_path = './best_model'
    
    import datetime
    
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(KST)
    now = now.strftime("%mM%dD%HH%MM")
    model_name = 'model_{}_{}'.format(model_name, now)

    # 경로와 이름을 합쳐서 완전한 경로 생성
    model_path = os.path.join(model_save_path, model_name)

    # 모델 저장 경로에 폴더가 없으면 폴더 생성
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # 모델 저장
    model.save_pretrained(model_path)
    
if __name__ == "__main__":
    main()