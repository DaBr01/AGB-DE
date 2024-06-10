from datasets import load_from_disk
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


ds = load_from_disk("../../loc_datasets/agb-de-under")

predictions = {}
predictedlbls = []

with open('../../responses/gpt35_responses.json', encoding='utf-8') as f:
    predictions = json.load(f)

predictions = predictions["responses"]

for item in ds["test"]:
    for x in predictions:
        if x['id'] == item['id']:
            predictedlbls.append(int(x['ungültig']))
            break


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ds["test"]["label"], predictedlbls)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ds["test"]["label"], predictedlbls)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ds["test"]["label"], predictedlbls)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ds["test"]["label"], predictedlbls)
print('F1 score: %f' % f1)

CM = confusion_matrix(ds["test"]["label"], predictedlbls)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print(CM)


print("tp: " + str(TP))
print("tn: " + str(TN))
print("fp: " + str(FP))
print("fn: " + str(FN))