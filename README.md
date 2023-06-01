# level2_nlp_datacentric-nlp-04
모델링 변경 없이 데이터를 중심으로 성능 향상을 목표로 합니다.

## 프로젝트 소개
연합뉴스의 뉴스 제목을 입력으로 받아 IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 총 7개의 클래스로 분류하는 프로젝트입니다.  
train.csv의 12%에는 g2p(grapheme to phoneme)가 적용되어 뉴스 제목에 노이즈가 있고, 3%에는 클래스가 무작위로 변경되는 노이즈가 있습니다.  
모델링 변경 없이 데이터를 중심으로 성능 향상을 목표로 합니다.

## 개발 기간
* 23.05.22 - 23.06.01(총 11일)

### 멤버 구성 및 역할
* 곽민석 : 
* 이인균 : 
* 임하림 : 
* 최휘민 : 
* 황윤기 : 

## 실행 방법


## 🎛️ Data Controll Center
### 1. 실행 방법
1. `main.py` 내에 `FILE_PATH` 변수를 설정 해주셔야 합니다. 
2. 다음의 명령어를 이용하여 dependency를 설치 해주셔야 합니다. 

```
pip install streamlit
pip install matplotlib
apt-get install fonts-nanum*
```

3. 다음 명령어를 이용하여 실행하시면 됩니다. 

```
streamlit run main.py --server.port PORT_NUMBER
```
### 2. 오류 발생시
#### 1. 폰트를 설치하였으나 폰트를 찾을 수 없다 하는 경우. 

- upstage 서버 기준으로 다음의 명령어를 실행한 후 다시 실행 시켜보시기 바랍니다. 

```
rm -rf /opt/ml/.cache/matplotlib
```

#### 2. "1. 실행 방법" 3번에서 실행되지 않는 경우. 

- 다음의 명령어를 이용하여 실행 시켜보시기 바랍니다. 

```
streamlit run main.py --server.port PORT_NUMBER --server.fileWatcherType none
```

### 3. Function
#### Easy Miss Label Filtering
![Miss Label Filtering](/Images/easy_miss_label_filter.gif)
- `Class Percentile` Value를 조절해서, Miss Label의 변경을 관찰하세요!
- `OOD(Out of Distribution)` 을 제외하고, Miss Label이 Filtering된 Data를 쉽게 다운로드하세요!
- 해당 기능을 사용하기 위해선, Miss Label Filtering을 시행할 데이터의 예측확률이 필요합니다.
  - 아래의 코드를 학습 코드 추가해주세요.
      - 맨 위 Directory Path 정의 구간
      ```python
      PREDICT_DIR = os.path.join(BASE_DIR, "prediction") if os.path.exists(os.path.join(BASE_DIR, "prediction")) else os.mkdir(os.path.join(BASE_DIR, "prediction"))
      ```
      - 맨 아래 학습이 종료된 후 구간
      ```python
      model.eval()
      preds = []
      for idx, sample in tqdm(dataset_train.iterrows(), # tqdm 사용 안할 시에 삭제
                              total=len(dataset_train),
                              desc='Predicting'):
          inputs = tokenizer(sample['text'],
                            max_length=128,
                            padding="max_length",
                            return_tensors="pt").to(DEVICE)
          with torch.no_grad():
              logits = model(**inputs).logits
              # pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
              pred = torch.nn.Softmax(dim=1)(logits).cpu().numpy()
              preds.extend(pred)

      dataset_train['preds_value'] = np.array(preds).tolist()
      dataset_train.to_csv(os.path.join(PREDICT_DIR, 'train_prediction.csv'), index=False)
      ```
      ```python
      model.eval()
      preds = []
      for idx, sample in tqdm(dataset_valid.iterrows(), # tqdm 사용 안할 시에 삭제
                              total=len(dataset_valid),
                              desc='Predicting'):
          inputs = tokenizer(sample['text'],
                            max_length=128,
                            padding="max_length",
                            return_tensors="pt").to(DEVICE)
          with torch.no_grad():
              logits = model(**inputs).logits
              # pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
              pred = torch.nn.Softmax(dim=1)(logits).cpu().numpy()
              preds.extend(pred)

      dataset_valid['preds_value'] = np.array(preds).tolist()
      dataset_valid.to_csv(os.path.join(PREDICT_DIR, 'valid_prediction.csv'), index=False)
      ```
