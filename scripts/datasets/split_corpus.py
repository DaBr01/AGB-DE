from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

test_size = 0.2

df_all = pd.read_csv("../../corpus/agb-de-anonym.csv")
df_all = df_all.rename(columns={"void": "label"})
df_all["topics"] = df_all["topics"].str.split(",")

# make sure split is stratified by topic and label
df_singlelabel = df_all[df_all['topics'].apply(lambda x: len(x) == 1)]
df_multilabel = df_all[df_all['topics'].apply(lambda x: len(x) > 1)]

df_singlelabel_0 = df_singlelabel[df_singlelabel['label'].apply(lambda x: x == 0.0)]
df_singlelabel_1 = df_singlelabel[df_singlelabel['label'].apply(lambda x: x == 1.0)]

# remove combinations of topic and label that only occure once
topic_counts = df_singlelabel_1['topics'].value_counts()
unique_topics = topic_counts[topic_counts == 1].index
unique_topics_df = df_singlelabel_1[df_singlelabel_1['topics'].isin(unique_topics)]
df_singlelabel_1 = df_singlelabel_1[~df_singlelabel_1['topics'].isin(unique_topics)]

df_train = [None, None]
df_test = [None, None]

df_train[0], df_test[0] = train_test_split(df_singlelabel_0, test_size=test_size, random_state=333,
                                           stratify=df_singlelabel_0['topics'])
df_train[1], df_test[1] = train_test_split(df_singlelabel_1, test_size=test_size, random_state=333,
                                           stratify=df_singlelabel_1['topics'])

for index, row in df_multilabel.iterrows():
    d_tr = df_train[int(row["label"])]
    d_te = df_test[int(row["label"])]
    train_count = len(d_tr.loc[d_tr.topics.apply(lambda x: row["topics"][0] in x)])
    test_count = len(d_te.loc[d_te.topics.apply(lambda x: row["topics"][0] in x)])

    if (train_count == 0):
        train_count = len(d_tr.loc[d_tr.topics.apply(lambda x: row["topics"][1] in x)])
        test_count = len(d_te.loc[d_te.topics.apply(lambda x: row["topics"][1] in x)])
        continue

    # stratify
    add_test_share = (test_count + 1) / (train_count + test_count + 1)
    add_train_share = test_count / (train_count + 1 + test_count)

    if abs(add_train_share - 0.2) <= abs(add_test_share - 0.2):
        df_train[int(row["label"])] = df_train[int(row["label"])]._append(row, ignore_index=True)
    else:
        df_test[int(row["label"])] = df_test[int(row["label"])]._append(row, ignore_index=True)

# create dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(pd.concat([df_train[0], df_train[1]], ignore_index=True, sort=False)),
    "test": Dataset.from_pandas(pd.concat([df_test[0], df_test[1]], ignore_index=True, sort=False)),
})

dataset = dataset.class_encode_column("label")


#dataset.save_to_disk('../../loc_datasets/agb-de')


# undersample dataset

def remove_items(df, topic, amount):
    ind = df[df.topics.apply(lambda x: x[0]) == topic].index
    drop = np.random.choice(ind, size=int(amount), replace=False)
    df = df.drop(drop)
    return df


for lbl in [0, 1]:
    df = pd.concat([df_train[lbl], df_test[lbl]], ignore_index=True, sort=False)

    remove = df.topics.apply(lambda x: x[0]).value_counts()

    for topic, count in remove.items():
        if not count - 100 > 0:
            continue

        remove_total = count - 100
        remove_test = round(0.2 * remove_total, 0)
        remove_train = remove_total - remove_test

        df_train[lbl] = remove_items(df_train[lbl], topic, remove_train)
        df_test[lbl] = remove_items(df_test[lbl], topic, remove_test)


dataset_under = DatasetDict({
    "train": Dataset.from_pandas(pd.concat([df_train[0], df_train[1]], ignore_index=True, sort=False)),
    "test": Dataset.from_pandas(pd.concat([df_test[0], df_test[1]], ignore_index=True, sort=False)),
})

dataset_under = dataset_under.class_encode_column("label")


dataset_under.save_to_disk('../../loc_datasets/agb-de-under')