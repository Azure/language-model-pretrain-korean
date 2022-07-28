from datasets import load_dataset, concatenate_datasets, Dataset
from Korpora import Korpora




## Function used to preprocess the text and split sentences
def split(example):
    sentence = []
    for text in example["text"]:
        sentences = [t.lstrip() if t.endswith(".") else t.lstrip() + "." for t in text.replace("\n", "").split(". ") if len(t) > 1]
        sentence += sentences
    return {"text": sentence}


## CC-100-Ko
cc100_ko = load_dataset("cc100", lang="ko")
cc100_ko = cc100_ko["train"].remove_columns("id")
cc100_ko = cc100_ko.filter(lambda example: len(example["text"]) >= 50)
cc100_ko = cc100_ko.map(split, batched=True, batch_size=1000, remove_columns=cc100_ko.column_names)
cc100_ko = cc100_ko.filter(lambda example: len(example["text"]) >= 50)
cc100_ko.to_json("./cc100_ko.json")


## Wiki-Ko
wiki_ko = load_dataset("text", data_files="./wiki_ko.txt")["train"]
wiki_ko = wiki_ko.filter(lambda example: len(example["text"]) >= 50)
wiki_ko = wiki_ko.map(split, batched=True, batch_size=1000, remove_columns=wiki_ko.column_names)
wiki_ko = wiki_ko.filter(lambda example: len(example["text"]) >= 50)
wiki_ko.to_json("./wiki_ko.json")


## NamuWiki
namuwikitext = Korpora.load("namuwikitext")
namuwikitext = Dataset.from_dict({"text": namuwikitext.get_all_texts()})
namuwikitext = namuwikitext.filter(lambda example: len(example["text"]) >= 50)
namuwikitext = namuwikitext.map(split, batched=True, batch_size=1000, remove_columns=namuwikitext.column_names)
namuwikitext = namuwikitext.filter(lambda example: len(example["text"]) >= 50)
namuwikitext.to_json("./namuwikitext.json")


## Petition
korean_petitions = Korpora.load("korean_petitions")
korean_petitions = Dataset.from_dict({"text": korean_petitions.get_all_texts()})
korean_petitions = korean_petitions.filter(lambda example: len(example["text"]) >= 50)
korean_petitions = korean_petitions.map(split, batched=True, batch_size=1000, remove_columns=korean_petitions.column_names)
korean_petitions = korean_petitions.filter(lambda example: len(example["text"]) >= 50)
korean_petitions.to_json("./korean_petitions.json")


## Base corpus
cc100_ko = load_dataset("json", data_files="./cc100_ko.json")["train"]
ko_corpus_base = cc100_ko.train_test_split(test_size=0.25, seed=2022)["test"]
ko_corpus_base.to_json("./ko_corpus_base.json")

with open("./ko_corpus_base.txt", "a", encoding="utf-8") as f:
    for i in range(len(ko_corpus_base)):
        text = ko_corpus_base[i]["text"]
        f.write(text)
        f.write('\n')


## Large corpus
cc100_ko = load_dataset("json", data_files="./cc100_ko.json")["train"]
wiki_ko = load_dataset("json", data_files="./wiki_ko.json")["train"]
namuwikitext = load_dataset("json", data_files="./namuwikitext.json")["train"]
korean_petitions = load_dataset("json", data_files="./korean_petitions.json")["train"]
ko_corpus_large = concatenate_datasets([cc100_ko, wiki_ko, namuwikitext, korean_petitions])
ko_corpus_large.to_json("./ko_corpus_large.json")

with open("./ko_corpus_large.txt", "a", encoding="utf-8") as f:
    for i in range(len(ko_corpus_large)):
        text = ko_corpus_large[i]["text"]
        f.write(text)
        f.write('\n')



