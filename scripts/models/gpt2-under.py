from datasets import Dataset
from datasets import load_from_disk
import pandas as pd
from sklearn.model_selection import train_test_split


import torch

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())

ds = load_from_disk("../../loc_datasets/agb-de-under")

savename = "agb-de-under-gpt2"
modelname = 'benjamin/gerpt2'


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(modelname)

#tokenizer.pad_token = tokenizer.eos_token


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenizer.model_max_length = 1024

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_data = ds.map(tokenize_function, batched=True)

from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    metric = evaluate.load("recall")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2)


from huggingface_hub import login

login("key")

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="../../models/temp",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

import torch
from transformers import Trainer
torch.cuda.empty_cache()

import gc
gc.collect()

import torch
from transformers import Trainer
torch.cuda.empty_cache()

import gc
gc.collect()


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        model.to(device)
        self.model.to(device)


        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0])).to('cuda:0')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("eval")
print(trainer.evaluate())

trainer.save_model("../../models/" + savename)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


prediction = trainer.predict(test_dataset=tokenized_data["test"])

#print(prediction)

predictedlbls = np.argmax(prediction.predictions,  axis=1)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(tokenized_data["test"]["label"], predictedlbls)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(tokenized_data["test"]["label"], predictedlbls)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(tokenized_data["test"]["label"], predictedlbls)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(tokenized_data["test"]["label"], predictedlbls)
print('F1 score: %f' % f1)

CM = confusion_matrix(tokenized_data["test"]["label"], predictedlbls)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print(CM)


print("tp: " + str(TP))
print("tn: " + str(TN))
print("fp: " + str(FP))
print("fn: " + str(FN))