# level2_nlp_datacentric-nlp-04

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