import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, 
from transformers import XLMProphetNetForConditionalGeneration, XLMProphetNetTokenizer




## Specify the task (qg)
task = "qg"


## Specify the model (ProphetNet-Ko_Base/ProphetNet-Ko_Large/ProphetNet-Multi_Large)
model_name = "ProphetNet-Ko_Base"

if model_name == "ProphetNet-Multi_Large":
    model = XLMProphetNetForConditionalGeneration.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
elif model_name == "ProphetNet-Ko_Base":
    model = XLMProphetNetForConditionalGeneration.from_pretrained("./base")
    tokenizer = XLMProphetNetTokenizer.from_pretrained("./base")
elif model_name == "ProphetNet-Ko_Large":
    model = XLMProphetNetForConditionalGeneration.from_pretrained("./large")
    tokenizer = XLMProphetNetTokenizer.from_pretrained("./large")


## Prepare the data
def prepare_data(batch):
    
    inputs = tokenizer.batch_encode_plus(batch["context"], truncation=True, max_length=512)
    labels = tokenizer.batch_encode_plus(batch["question"], truncation=True, max_length=64)

    batch["input_ids"] = inputs.input_ids.copy()
    batch["attention_mask"] = inputs.attention_mask.copy()
    batch["decoder_input_ids"] = labels.input_ids.copy()
    batch["decoder_attention_mask"] = labels.attention_mask.copy()

    return batch

if task == "qg":
    finetune_data = load_dataset("json", data_files={"train": "./ko_qg_train.json", "eval": "./ko_qg_eval.json"})
    finetune_data = finetune_data.map(prepare_data, batched=True, batch_size=1000)


## Data collator for question generation
class ProphetNetDataCollatorForQG(DataCollatorForSeq2Seq):
        
    def __call__(self, batch):
        input_ids_batch = []
        input_ids_len = []
        
        labels_batch = []
        labels_len = []
        
        for example in batch:
            input_ids = example["input_ids"]
            input_ids_len.append(len(input_ids))
            input_ids_batch.append(input_ids)
            
            labels = example["decoder_input_ids"]
            labels_len.append(len(labels))
            labels_batch.append(labels)

        input_ids_padded = self.process_encoded_text(input_ids_batch, input_ids_len, self.tokenizer.pad_token_id)
        labels_padded = self.process_encoded_text(labels_batch, labels_len, self.label_pad_token_id)
        
        attention_mask = self.generate_attention_mask(input_ids_padded, self.tokenizer.pad_token_id)
        decoder_attention_mask = self.generate_attention_mask(labels_padded, self.tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels_padded,
        }

    def process_encoded_text(self, sequences, sequences_len, pad_token_id):
        sequences_max_len = np.max(sequences_len)
        max_length = min(sequences_max_len, self.max_length)
        padded_sequences = self.pad_sequences(sequences, max_length, pad_token_id)
        return torch.LongTensor(padded_sequences)

    def generate_attention_mask(self, input_ids, pad_token_id):
        return (input_ids != pad_token_id).long()
    
    def pad_sequences(self, sequences, max_length, pad_token_id):
        num_samples = len(sequences)
        padded_sequences = np.full((num_samples, max_length), pad_token_id)
        for i, sequence in enumerate(sequences):
            sequence = np.array(sequence)[:max_length]
            padded_sequences[i, :len(sequence)] = sequence
        return padded_sequences


## Fine-tune
data_collator = ProphetNetDataCollatorForQG(tokenizer=tokenizer, max_length=512)

training_args = Seq2SeqTrainingArguments(
    output_dir="./"+task+"/"+model_name,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
    num_train_epochs=10,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    logging_strategy="epoch",
    save_strategy="epoch"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=finetune_data["train"],
    eval_dataset=finetune_data["eval"],
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("./"+task+"/"+model_name)



