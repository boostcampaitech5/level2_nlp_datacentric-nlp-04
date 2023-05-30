import torch
import pandas as pd
import numpy as np

def ood_cls_filter(dataset:pd.DataFrame, preds: list, ood_percent=0.5, class_percent=80):
	"""
	Params:
		dataset: 원본 데이터셋
		preds: 모델의 예측 확률(Softmax 값)
		ood_percent: OOD Threshold의 백분위
		class_percent: Class Threshold의 백분위

	Returns:
		dataset_cls: Class Threshold 이상으로 분류된 데이터셋
		dataset_ood: OOD Threshold 이하로 분류된 데이터셋
		distribution: 각 클래스의 OOD Threshold, Class Threshold
	"""
	ood_value = []
	class_value = []
	idx_value = []
	distribution = pd.DataFrame(columns=['Class', 'OOD Threshold', 'Class Threshold'])
	num_to_class = {0: 'IT과학',
					1: '경제',
					2: '사회',
					3: '생활문화',
					4: '세계',
					5: '스포츠',
					6: '정치'}

	preds = preds.apply(lambda x : eval(x))
	eval_preds = torch.tensor(preds)
	eval_indices = torch.argmax(eval_preds, dim=-1)
	dataset['pred'] = eval_indices
	dataset = dataset.reset_index(drop=True)

	for i in range(7):
		idx_value.append(eval_preds[torch.where(eval_indices == i)][:, i].sort().values)

	for i in range(7):
		ood_value.append(idx_value[i][int(len(idx_value[i]) * ood_percent / 100)].tolist())
		class_value.append(idx_value[i][int(len(idx_value[i]) * class_percent / 100)].tolist())
		distribution = pd.concat([distribution, pd.DataFrame({"Class": num_to_class[i],
															  "OOD Threshold": ood_value[i],
															  "Class Threshold": class_value[i]},index=[0])],
								 ignore_index=True)

	ood_value = torch.tensor(ood_value)
	class_value = torch.tensor(class_value)

	check_ood = eval_preds < ood_value
	check_class = eval_preds > class_value
	ood_idx = torch.where(torch.all(check_ood, dim=1) == True)[0].tolist()
	class_idx = torch.where(torch.count_nonzero(check_class, dim=1) == 1)[0].tolist()

	dataset_ood = dataset.iloc[ood_idx]
	dataset_cls = dataset.iloc[class_idx]

	issue_ratio = dataset_cls[dataset_cls['target'] != dataset_cls['pred']].groupby(['pred'])['ID'].count() / \
				  dataset_cls.groupby(['pred'])['ID'].count()

	distribution = pd.concat([distribution,
							  issue_ratio.rename("Issue Ratio")],
							 axis=1)

	return dataset_cls, dataset_ood, distribution
