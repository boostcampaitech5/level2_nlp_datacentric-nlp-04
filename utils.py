import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
	"""
	Confusion Matrix를 시각화하는 함수(* Seaborn 설치 필요)
	!! 한글 폰트 설치가 필요합니다
		1. apt-get install fonts-nanum*
		2. cp /usr/share/fonts/truetype/nanum/Nanum* /opt/conda/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/
		3. rm -rf ~/.cache/matplotlib/*
	Params:
		y_true: True Label
		y_pred: Predicion Label

	"""
	label_to_num = {'정치': 0,
					'경제': 1,
					'사회': 2,
					'생활문화': 3,
					'세계': 4,
					'IT과학': 5,
					'스포츠': 6}
	plot = confusion_matrix(y_true, y_pred, normalize='true')
	plt.rcParams['font.family'] = "NanumGothicCoding"

	fig, ax = plt.subplots(figsize=(15, 15))
	sns.heatmap(plot, annot=True, fmt='.2f', cmap="coolwarm",
				xticklabels=list(label_to_num.keys()),
				yticklabels=list(label_to_num.keys()),
				annot_kws={"fontsize": "xx-large"})
	plt.ylabel('True Label')
	plt.xlabel("Predicted Label")

	return fig