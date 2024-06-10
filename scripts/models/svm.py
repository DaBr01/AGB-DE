from datasets import load_from_disk
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

ds = load_from_disk("../../loc_datasets/agb-de-under")

vectorizer = CountVectorizer()
tfidf_train_vectors = vectorizer.fit_transform(ds["train"]["text"])
tfidf_test_vectors = vectorizer.transform(ds["test"]["text"])

classifier = svm.SVC(kernel='linear')
classifier.fit(tfidf_train_vectors, ds["train"]["label"])
y_pred = classifier.predict(tfidf_test_vectors)


out = {"responses":[]}
resname = "../../responses/svm-under.json"

for i in range(len(y_pred)):
    entry = {"id":ds["test"]["id"][i], "ungültig": bool(y_pred[i] == 1)}
    out["responses"].append(entry)

with open(resname, "w", encoding="utf-8") as file:
    json.dump(out, file, indent=4, ensure_ascii=False)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ds["test"]["label"], y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ds["test"]["label"], y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ds["test"]["label"], y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ds["test"]["label"], y_pred)
print('F1 score: %f' % f1)

CM = confusion_matrix(ds["test"]["label"], y_pred)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print(CM)
