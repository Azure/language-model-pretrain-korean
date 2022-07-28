import numpy as np
from datasets import load_dataset
# from transformers import ProphetNetTokenizer
from transformers import XLMProphetNetTokenizer




## Function used to prepare the pretrain data for ProphetNet
def prepare_data(batch, s=64, k=10, shift=1):
    inputs = tokenizer(batch["text"], truncation=True, max_length=512)
    batch["input_ids"] = inputs.input_ids.copy()
    batch["attention_mask"] = inputs.attention_mask.copy()
    batch["decoder_input_ids"] = inputs.input_ids.copy()
    batch["decoder_attention_mask"] = inputs.attention_mask.copy()
    batch["labels"] = inputs.input_ids.copy()
    batch["labels_ngram"] = inputs.input_ids.copy()
    
    for i in range(len(batch["text"])):
        ntokens = len(batch["input_ids"][i]) - 1
        if ntokens > 20 and ntokens < s:
            ids = [np.random.randint(1, ntokens-k-shift)]
        elif ntokens >= s and ntokens < 2*s:
            if ntokens-k-shift > s:
                ids = np.random.randint(np.append(1,np.arange(1,2)*s), np.append(np.arange(1,2)*s-k-shift,ntokens-k-shift))
            else:
                ids = [np.random.randint(1, s-k-shift)]
        elif ntokens >= 2*s and ntokens < 3*s:
            if ntokens-k-shift > 2*s:
                ids = np.random.randint(np.append(1,np.arange(1,3)*s), np.append(np.arange(1,3)*s-k-shift,ntokens-k-shift))
            else:
                ids = np.random.randint(np.append(1,np.arange(1,2)*s), np.arange(1,3)*s-k-shift)
        elif ntokens >= 3*s and ntokens < 4*s:
            if ntokens-k-shift > 3*s:
                ids = np.random.randint(np.append(1,np.arange(1,4)*s), np.append(np.arange(1,4)*s-k-shift,ntokens-k-shift))
            else:
                ids = np.random.randint(np.append(1,np.arange(1,3)*s), np.arange(1,4)*s-k-shift)
        elif ntokens >= 4*s and ntokens < 5*s:
            if ntokens-k-shift > 4*s:
                ids = np.random.randint(np.append(1,np.arange(1,5)*s), np.append(np.arange(1,5)*s-k-shift,ntokens-k-shift))
            else:
                ids = np.random.randint(np.append(1,np.arange(1,4)*s), np.arange(1,5)*s-k-shift)
        elif ntokens >= 5*s and ntokens < 6*s:
            if ntokens-k-shift > 5*s:
                ids = np.random.randint(np.append(1,np.arange(1,6)*s), np.append(np.arange(1,6)*s-k-shift,ntokens-k-shift))
            else:
                ids = np.random.randint(np.append(1,np.arange(1,5)*s), np.arange(1,6)*s-k-shift)
        elif ntokens >= 6*s and ntokens < 7*s:
            if ntokens-k-shift > 6*s:
                ids = np.random.randint(np.append(1,np.arange(1,7)*s), np.append(np.arange(1,7)*s-k-shift,ntokens-k-shift))
            else:
                ids = np.random.randint(np.append(1,np.arange(1,6)*s), np.arange(1,7)*s-k-shift)
        elif ntokens >= 7*s:
            if ntokens-k-shift > 7*s:
                ids = np.random.randint(np.append(1,np.arange(1,8)*s), np.append(np.arange(1,8)*s-k-shift,ntokens-k-shift))
            else:
                ids = np.random.randint(np.append(1,np.arange(1,7)*s), np.arange(1,8)*s-k-shift)
        else:
            ids = None
            
        batch["decoder_input_ids"][i] = []
        batch["decoder_attention_mask"][i] = []
        batch["labels"][i] = []
        batch["labels_ngram"][i] = []
        
        if ids is not None:
            for j in ids:
                batch["decoder_input_ids"][i] += (batch["input_ids"][i][(j-1):(j+k-1)]).copy()
                batch["decoder_attention_mask"][i] += [1] * k
                batch["labels"][i] += (batch["input_ids"][i][(j):(j+k)]).copy()
                batch["labels_ngram"][i] += (batch["input_ids"][i][(j+1):(j+k+1)]).copy()
                
                for l in range(k):
                    if np.random.binomial(n=1,p=0.8,size=1)[0] == 1:
                        batch["input_ids"][i][j+l] = tokenizer.mask_token_id
                    elif np.random.binomial(n=1,p=0.5,size=1)[0] == 1:
                        batch["input_ids"][i][j+l] = np.random.randint(20, tokenizer.vocab_size)
    
    return batch
    

## Base
tokenizer = XLMProphetNetTokenizer("./ko_tokenizer_base.model", model_max_length=512)
ko_corpus_base = load_dataset("json", data_files="./ko_corpus_base.json")["train"]
pretrain_data_base = ko_corpus_base.map(prepare_data, batched=True, batch_size=1000)
pretrain_data_base = pretrain_data_base.filter(lambda example: len(example["decoder_input_ids"]) > 0)
pretrain_data_base.to_json("./pretrain_data_base.json")
pretrain_data_base_compressed = pretrain_data_base.remove_columns(["text", "attention_mask", "decoder_attention_mask"])
pretrain_data_base_compressed.to_json("./pretrain_data_base_compressed.json")


## Large
tokenizer = XLMProphetNetTokenizer("./ko_tokenizer_large.model", model_max_length=512)
ko_corpus_large = load_dataset("json", data_files="./ko_corpus_large.json")["train"]
pretrain_data_large = ko_corpus_large.map(prepare_data, batched=True, batch_size=1000)
pretrain_data_large = pretrain_data_large.filter(lambda example: len(example["decoder_input_ids"]) > 0)
pretrain_data_large.to_json("./pretrain_data_large.json")
pretrain_data_large_compressed = pretrain_data_large.remove_columns(["text", "attention_mask", "decoder_attention_mask"])
pretrain_data_large_compressed.to_json("./pretrain_data_large_compressed.json")



