# AGB-DE: A Corpus for the Automated Legal Assessment of Clauses in German Consumer Contracts

AGB-DE is a legal NLP corpus for the automated detection of potentially void clauses in German standard form consumer contracts. It consists of 3,764 clauses that have been legally assessed by experts and annotated as potentially void (1) or valid (0). Additionally, each clause is annotated with a topic label. This repository contains the corpus itself, code that was uses to anonymize the data, code that was used to train and evaluate baseline models, and the results of the baseline evaluation itself.

## How to cite
```
@inproceedings{braun-matthes-2024-agb,
    title = "AGB-DE: A Corpus for the Automated Legal Assessment of Clauses in German Consumer Contracts", 
    author = "Braun, Daniel and Matthes, Florian",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2024",
    publisher = "Association for Computational Linguistics"
}
```

## Data

The dataset consists of 3,764 clauses from 93 German consumer standard form contracts. Each clause has been annotated with its topic(s) and whether the clause is valid (0) or potentially void (1). The table below shows the distribution of topics among the clauses and the share of potentially void clauses per topic. For more details about the data please take a look at the [paper](https://arxiv.org/abs/2406.06809) and the [datasheet](https://github.com/DaBr01/AGB-DE/wiki).

| topic |	number of clauses |	share_void|
|-------|-----------------|----------|
|age| 	20	| 0|
|applicability	|148	|2.03|
|applicableLaw|	87|	3.45|
|arbitration|	97|	1.03|
|changes	|9	|11.11|
|codeOfConduct|	29	|0|
|conclusionOfContract|	557	|5.92|
|contractLanguage|	41	|0|
|delivery|	475	|7.16|
|description|	46	|0|
|disposal|	36	|0|
|intellectualProperty|	39	|0|
|language|	9	|11.11|
|liability	|211	|9|
|party|	0	|0|
|payment|	642	|6.07|
|personalData|	115	|0.87|
|placeOfJurisdiction|	53|	1.89|
|prices|	147	|1.36|
|retentionOfTitle|	125	|2.4|
|severability	|35	|11.43|
|textStorage|	57	|1.75|
|warranty|	314	|6.37|
|withdrawal	|506	|3.75|
|Total lvl 1|	3798	|4.8|


## Baseline Automated Legal Assessment

### ``agb-de`` dataset
| Model                      | Precision | Recall   | F1-Score |
|----------------------------|----------|----------|----------|
| ``svm``                    | 0.37     | 0.27     | 0.31     |
| ``bert-base-german-cased`` | 0.50     | 0.27     | **0.35** |
| ``xlm-roberta-base``       | 0.00     | 0.00     | 0.00     |
| ``gerpt2``                 | **0.71** | 0.14     | 0.23     |
| ``gpt-3.5-turbo-0125 ``    | 0.06     | **0.92** | 0.11     |


### ``agb-de-under`` dataset
| Model                     | Precision | Recall   | F1-Score |
|---------------------------|-----------|----------|----------|
| ``svm``                   | 0.40      | 0.32     | 0.36     |
| ``bert-base-german-cased``| 0.51      | 0.57     | **0.54** |
| ``xlm-roberta-base``      | **0.75**  | 0.08     | 0.15     |
| ``gerpt2``                | 0.64      | 0.43     | 0.52     |
| ``gpt-3.5-turbo-0125 ``   | 0.13      | **0.92** | 0.22     |

## How to use

The easiest way to use the corpus is through the [dataset](https://huggingface.co/datasets/d4br4/agb-de) and [baseline](https://huggingface.co/d4br4/AGBert) model provided on 🤗 Huggingface. See also [example.py](usage/example.py).

```python
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
```

## Contact

If you have any question, please contact:

[Daniel Braun](https://www.daniel-braun.science) (University of Twente)<br>
[d.braun@utwente.nl](mailto:d.braun@utwente.nl)

## Acknowledgment
The data collection and annotation was supported by funds of the Federal Ministry of Justice and Consumer
Protection (BMJV) based on a decision of the Parliament of the Federal Republic of Germany
via the Federal Office for Agriculture and Food (BLE) under the innovation support
programme.
