import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Font Setting
# 작동하지 않는 경우 아래의 명령어를 이용하여를 사용하여 설치. 
# apt-get install fonts-nanum*
# 설치하여도 실행 시 오류가 발생하는 경우 다음 명령어 실행. 
# rm -rf /opt/ml/.cache/matplotlib
plt.rcParams['font.family'] = 'NanumGothic'

# 원본 train file. 
FILE_PATH = "/opt/ml/data/custom_data/origin_train.csv"
assert FILE_PATH != "", "Please set FILE_PATH. "

df = pd.read_csv(FILE_PATH)

st.markdown("# 🎛️ Data Control Center")
st.markdown("## 원본 데이터")
st.dataframe(df.head(20))

st.markdown("### 데이터 label 비율")
label_counts = df['label_text'].value_counts()

# 비율 차트 생성
fig, ax = plt.subplots(figsize=(8, 6))
label_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_title('원본 데이터 비율')

# streamlit을 통해 차트 표시
st.pyplot(fig)

st.markdown("## 🕹️ 데이터 조작")
st.markdown("### 중복 제거")
options = st.multiselect(
    '중복 제거 옵션을 선택하세요. ',
    ['ID', 'input_text'],
    ['ID', 'input_text'])

for option in options:
    df = df.drop_duplicates(subset=option, keep=False)

st.markdown("## 변경 후")
df

st.markdown("## 변경 후")
label_counts = df['label_text'].value_counts()

fig, ax = plt.subplots(figsize=(8, 6))
label_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_title('작업 후 데이터 비율')

st.pyplot(fig)

# 수정된 CSV 파일 다운로드. 
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df)

st.download_button(
   "📥 Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)