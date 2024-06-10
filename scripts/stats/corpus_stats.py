# calculate corpus statistics

import pandas as pd


# base corpus
df = pd.read_csv("../../corpus/agb-de-anonym.csv")


# topic distribution
df["topics"] = df["topics"].str.split(",")
df["subtopics"] = df["subtopics"].fillna('')
df["subtopics"] = df["subtopics"].str.split(",")

stats = pd.concat(
    [pd.Series([item for sublist in df["topics"] for item in sublist]).value_counts(),
     pd.Series([item for sublist in df["subtopics"] for item in sublist]).value_counts()]
)

stats = stats.drop(index="")
stats = stats.sort_index().to_frame()
stats.columns = ["Amount"]
stats = stats.rename_axis("Label").reset_index()


# void share
stats["Share_Void"] = stats.Label.apply(lambda y: "{:.2f}".format(df.loc[df['topics'].apply(lambda x: y in x), 'void'].sum() / len(df.loc[df['topics'].apply(lambda x: y in x)]) * 100))
stats["Share_Void"] = stats["Share_Void"].apply(lambda y: "0.00" if y == "nan" else y)


# sum
stats.loc[len(stats.index)] = ['Total lvl 1', stats.loc[~stats['Label'].str.contains(":"), 'Amount'].sum(), "{:.2f}".format(df.void.sum() / len(df) * 100)]
stats.loc[len(stats.index)] = ['Total lvl 2', stats.loc[stats['Label'].str.contains(":"), 'Amount'].sum(), ""]


print(stats.to_string())
