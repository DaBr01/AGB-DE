from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

# load corpus
ds = load_dataset("d4br4/agb-de")

# load model
modelname = "d4br4/AGBert"
model = AutoModelForSequenceClassification.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)

# create classification pipeline
clf = pipeline("text-classification", model, tokenizer=tokenizer, max_length=512, truncation=True)

# classify clause text
prediction = clf.predict(ds["test"][0]["text"])

# check classification output
if prediction[0]["label"] == "valid":
    print("This clause is valid.")

else:
    print("This clause is void.")