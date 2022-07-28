import random
from datasets import load_dataset, concatenate_datasets




## Load original datasets
squad_kor_v1 = load_dataset("squad_kor_v1")
klue_mrc = load_dataset("klue", "mrc")


## Preprocess (unique context only appears in one split)
squad_kor_v1_context_train = list(set(squad_kor_v1["train"]["context"]))
squad_kor_v1_context_validation = list(set(squad_kor_v1["validation"]["context"]))
squad_kor_v1_context = list(set(squad_kor_v1_context_train+squad_kor_v1_context_validation))
klue_mrc_context_train = list(set(klue_mrc["train"]["context"]))
klue_mrc_context_validation = list(set(klue_mrc["validation"]["context"]))
klue_mrc_context = list(set(klue_mrc_context_train+klue_mrc_context_validation))
all_context = list(set(squad_kor_v1_context+klue_mrc_context))

ko_qg_squad_kor_v1 = concatenate_datasets([squad_kor_v1["train"], squad_kor_v1["validation"]])
ko_qg_klue_mrc = concatenate_datasets([klue_mrc["train"], klue_mrc["validation"]])

squad_kor_v1_context_id = []
for i in range(len(ko_qg_squad_kor_v1)):
    context = ko_qg_squad_kor_v1[i]["context"]
    context_id = all_context.index(context)
    squad_kor_v1_context_id.append(context_id)

klue_mrc_context_id = []
for i in range(len(ko_qg_klue_mrc)):
    context = ko_qg_klue_mrc[i]["context"]
    context_id = all_context.index(context)
    klue_mrc_context_id.append(context_id)

ko_qg_squad_kor_v1 = ko_qg_squad_kor_v1.add_column("context_id", squad_kor_v1_context_id)
ko_qg_klue_mrc = ko_qg_klue_mrc.add_column("context_id", klue_mrc_context_id)

ko_qg_squad_kor_v1 = ko_qg_squad_kor_v1.remove_columns(["id", "title", "answers"])
ko_qg_klue_mrc = ko_qg_klue_mrc.remove_columns(["title", "news_category", "source", "guid", "is_impossible", "question_type", "answers"])

ko_qg = concatenate_datasets([ko_qg_squad_kor_v1, ko_qg_klue_mrc])
ko_qg.to_json("./ko_qg.json")


## Split train and eval sets
index_eval = random.sample(range(len(all_context)), round(len(all_context)*0.1))
ko_qg_train = ko_qg.filter(lambda example: example["context_id"] not in index_eval)
ko_qg_eval = ko_qg.filter(lambda example: example["context_id"] in index_eval)
ko_qg_train.to_json("./ko_qg_train.json")
ko_qg_eval.to_json("./ko_qg_eval.json")



