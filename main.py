import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot_confusion_matrix
from filtering import ood_cls_filter, synthetic_filter


def correct_ood_cls():
    if st.session_state.ood_percent > st.session_state.cls_percent:
        st.session_state.cls_percent = st.session_state.ood_percent


@st.cache_data
def load_data(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
    return df


@st.cache_data
def process_label_issue(df_cls):
    return df_cls.drop(df_cls[df_cls['pred'] != df_cls['target']].index).to_csv(index=False, encoding='utf-8')

# 수정된 CSV 파일 다운로드.
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def show_control_page(FILE_PATH, split_name):
    # TODO: Change to Session State
    df = pd.read_csv(FILE_PATH)
    df = df.loc[:, ['ID', 'text', 'target', 'preds_value']]


    st.markdown("# 🎛️ Data Control Center")
    st.markdown("## 원본 데이터")
    st.dataframe(df.head(20))

    st.markdown("## 🕹️ 데이터 조작")

    # OOD, Class Filtering
    st.markdown("## 🪄 데이터 필터링")
    col1, col2 = st.columns([4, 6])


    dataset_cls, dataset_ood, distribution = ood_cls_filter(df, df['preds_value'], st.session_state['ood_percent'], st.session_state['cls_percent'])
    st.session_state['distribution'+split_name] = distribution
    with col2:
        st.pyplot(plot_confusion_matrix(dataset_cls['target'], dataset_cls['pred']))
        st.download_button("📥 Download Processed Label Issue Data CSV",
                           process_label_issue(dataset_cls),
                           "processed_dataset.csv",
                           "text/csv",
                           key='download-label-issue-csv')

    with col1:
        st.write("Numbers of Class Data: ", len(dataset_cls))
        st.dataframe(st.session_state['distribution'+split_name])
        st.markdown("---")
        st.markdown("### Label Issue")
        st.info("Total Label Issue Numbers : " + str(len(dataset_cls[dataset_cls['pred'] != dataset_cls['target']])))
        with st.container():
            option = st.selectbox("Select Label", list(st.session_state['distribution'+split_name]['Class'].unique()))
            label_issue_df = dataset_cls[(dataset_cls['pred'] != dataset_cls['target']) & (dataset_cls['target'] == label_to_num[option])].copy()
            st.write("Numbers of Label Issue Data :", len(label_issue_df))
            st.dataframe(label_issue_df[['text', 'target', 'pred']])

        for option in list(st.session_state['distribution'+split_name]['Class'].unique()):
            down_label_issue_df = dataset_cls[(dataset_cls['pred'] != dataset_cls['target']) & (dataset_cls['target'] == label_to_num[option])].copy()
            down_label_issue_df['target'] = down_label_issue_df['pred']
            df.iloc[down_label_issue_df.index, :] = down_label_issue_df
        st.download_button("📥 Download Label Issue Change Data CSV",
                             convert_df(df),
                            "label_issue_change.csv",
                            "text/csv",
                            key='download-label-issue-change-csv')


    col3, col4 = st.columns([5, 5])
    with col3:
        st.subheader("원본 데이터 label 비율")
        label_counts = df['target'].value_counts()
        label_counts.index = [num_to_label[i] for i in label_counts.index]

        # 비율 차트 생성
        fig, ax = plt.subplots(figsize=(8, 6))
        label_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_title('원본 데이터 비율')

        # streamlit을 통해 차트 표시
        st.pyplot(fig)

    with col4:
        st.subheader("작업 후 데이터 label 비율")
        mod_counts = dataset_cls[dataset_cls['pred'] == dataset_cls['target']]['target'].value_counts()
        mod_counts.index = [num_to_label[i] for i in mod_counts.index]

        fig, ax = plt.subplots(figsize=(8, 6))
        mod_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_title('작업 후 데이터 비율')

        st.pyplot(fig)



    csv = convert_df(df)
    st.download_button(
           "📥 Download",
           csv,
           "file.csv",
           "text/csv",
           key='download-csv'
        )

def show_synthetic_page(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
    df = df.loc[:, ['text', 'target', 'preds_value']]

    st.markdown("# 🧪 Synthetic Data Filtering")
    st.dataframe(df.head(20))

    col1, col2 = st.columns([5, 5])
    with col1:
        st.markdown("### Training Data Class Threshold")
        st.write(st.session_state['distribution'+"Train Data"]["Class Threshold"])
    with col2:
        st.markdown("### Validation Data Class Threshold")
        st.write(st.session_state['distribution'+"Validation Data"]["Class Threshold"])

    tab1, tab2 = st.tabs(['Train Class Threshold', 'Validation Class Threshold'])
    with tab1:
        train_correct_df = synthetic_filter(df, st.session_state['distribution' + "Train Data"]["Class Threshold"])
        st.write("Total Numbers : ", len(train_correct_df))
        st.dataframe(train_correct_df.head(20))
        st.download_button("📥 Download Filtered Synthetic Data Over Train Class Threshold",
                            convert_df(train_correct_df),
                            "filtered_synthetic_train.csv",
                            "text/csv",
                            key='download-synthetic-train-csv')

    with tab2:
        validation_correct_df = synthetic_filter(df, st.session_state['distribution' + "Validation Data"]["Class Threshold"])
        st.write("Total Numbers : ", len(validation_correct_df))
        st.dataframe(validation_correct_df.head(20))
        st.download_button("📥 Download Filtered Synthetic Data Over Validation Class Threshold",
                            convert_df(validation_correct_df),
                            "filtered_synthetic_validation.csv",
                            "text/csv",
                            key='download-synthetic-validation-csv')


st.set_page_config(layout="wide")
label_to_num = {'IT과학': 0,
                '경제': 1,
                '사회': 2,
                '생활문화': 3,
                '세계': 4,
                '스포츠': 5,
                '정치': 6}
num_to_label = {0: 'IT과학',
                1: '경제',
                2: '사회',
                3: '생활문화',
                4: '세계',
                5: '스포츠',
                6: '정치'}

# Font Setting
# 작동하지 않는 경우 아래의 명령어를 이용하여를 사용하여 설치.
# apt-get install fonts-nanum*
# 설치하여도 실행 시 오류가 발생하는 경우 다음 명령어 실행.
# rm -rf /opt/ml/.cache/matplotlib
plt.rcParams['font.family'] = 'NanumGothic'

with st.sidebar:
    st.header("Configuration")
    data_splits = st.selectbox("Choose Data Splits", ["Train Data", "Validation Data", "Synthetic Data"])
    ood_percent = st.slider("Out of Distribution Percentile", min_value=0.0, max_value=100.0, value=10.0, step=0.5,
                            key='ood_percent', on_change=correct_ood_cls)
    cls_percent = st.slider("Class Percentile", min_value=0.0, max_value=100.0, value=90.0, step=0.5,
                            key='cls_percent', on_change=correct_ood_cls)

    for split in ["Train Data", "Validation Data", "Synthetic Data"]:
        if split == "Train Data" and "FILE PATH"+split not in st.session_state:
            st.session_state["FILE PATH"+split] = "/opt/ml/prediction/train_prediction.csv"
            assert st.session_state["FILE PATH"+split] != "", "Please set FILE_PATH. "
        elif split == "Validation Data" and "FILE PATH"+split not in st.session_state:
            st.session_state["FILE PATH"+split] = "/opt/ml/prediction/valid_prediction.csv"
            assert st.session_state["FILE PATH"+split] != "", "Please set FILE_PATH. "
        elif split == "Synthetic Data" and "FILE PATH"+split not in st.session_state:
            st.session_state["FILE PATH"+split] = "/opt/ml/prediction/gpt_generated_prediction.csv"
            assert st.session_state["FILE PATH"+split] != "", "Please set FILE_PATH. "

        if split != "Synthetic Data":
            st.session_state["df" + split] = load_data(st.session_state["FILE PATH" + split])
            st.session_state["dataset_cls" + split], st.session_state["dataset_ood" + split], st.session_state["distribution" + split] = ood_cls_filter(st.session_state["df" + split],
                                                                                                                                                     st.session_state["df" + split]['preds_value'],
                                                                                                                                                     st.session_state['ood_percent'],
                                                                                                                                                     st.session_state['cls_percent'])

if data_splits == "Train Data" or data_splits == "Validation Data":
    show_control_page(st.session_state["FILE PATH"+data_splits], data_splits)
elif data_splits == "Synthetic Data":
    show_synthetic_page(st.session_state["FILE PATH"+data_splits])